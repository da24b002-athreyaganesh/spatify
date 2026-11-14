"""The Spatify gateway: the REST API the frontend talks to (interactive docs at /docs).

Listener endpoints:  POST /recommend, POST /feedback, GET /tracks/search
Operator endpoints:  GET /models, POST /models/rollback*, GET /pipeline/status,
                     GET /pipeline/runs, POST /pipeline/trigger*, GET /drift
Service endpoints:   GET /health (503 until Postgres and the model server answer), GET /metrics
* needs the admin key in the X-API-Key header.

The frontend reaches the model, the registry and the database only through these
endpoints (the rubric's "frontend and backend connected only through REST").

Run (as the Compose service does):
    uvicorn spatify.gateway.app:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import threading
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import psycopg
import psycopg_pool
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from mlflow.exceptions import MlflowException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from spatify.gateway.logs import setup_logging
from spatify.gateway.metrics import (
    CHAMPION,
    FEEDBACK,
    IN_PROGRESS,
    MODEL_UP,
    RECOMMENDATIONS,
    REQUEST_SECONDS,
    REQUESTS,
)
from spatify.gateway.model_client import (
    ModelClient,
    ModelServerUnavailableError,
    QueryRejectedError,
)
from spatify.gateway.registry import Registry, RegistryError
from spatify.gateway.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    Health,
    ModelsResponse,
    ModelVersion,
    PipelineRun,
    PipelineStatus,
    Recommendation,
    RecommendRequest,
    RecommendResponse,
    RollbackRequest,
    RollbackResponse,
    SearchResponse,
    Track,
    TriggerRequest,
)
from spatify.gateway.settings import Settings
from spatify.gateway.store import Store

log = logging.getLogger("spatify.gateway")

QUIET = {"/metrics", "/health"}  # polled every few seconds; logged at DEBUG only


def _text(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _track(row: dict) -> dict:
    """A catalogue or model-output row as the API's Track fields."""
    genres = _text(row.get("genres")) or ""
    return {
        "track_id": str(row["track_id"]),
        "track_name": _text(row.get("track_name")),
        "artists": _text(row.get("artists")),
        "album_name": _text(row.get("album_name")),
        "genres": [g for g in genres.split(";") if g],
        "popularity": int(row.get("popularity") or 0),
    }


def create_app(
    settings: Settings | None = None,
    store: Store | None = None,
    registry: Registry | None = None,
    model: ModelClient | None = None,
) -> FastAPI:
    """Build the app. Tests pass fakes for the store, registry and model client."""
    settings = settings or Settings.from_env()
    setup_logging(settings.log_level)
    store = store if store is not None else Store(settings.database_url)
    registry = registry or Registry(settings.mlflow_tracking_uri, settings.model_name)
    model = model or ModelClient(settings.model_server_url, settings.model_timeout_seconds)
    stop = threading.Event()

    def watch() -> None:
        """Every refresh_seconds: probe the model server, re-read the champion."""
        while True:
            MODEL_UP.set(1 if model.ping() else 0)
            try:
                registry.refresh()
                if registry.champion_version is not None:
                    CHAMPION.info({"version": str(registry.champion_version)})
            except Exception:
                log.warning("Couldn't refresh the champion from MLflow", exc_info=True)
            if stop.wait(settings.refresh_seconds):
                return

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.open()
        watcher = threading.Thread(target=watch, name="watch", daemon=True)
        watcher.start()
        log.info("Gateway started; model server %s", settings.model_server_url)
        yield
        stop.set()
        watcher.join(timeout=5)
        model.close()
        store.close()

    app = FastAPI(
        title="Spatify gateway",
        version="0.1.0",
        description="Cold-start music recommendations, feedback and model operations.",
        lifespan=lifespan,
    )

    def require_api_key(x_api_key: str | None = Header(None)) -> None:
        """Admin actions need the key whose SHA-256 is in API_KEY_SHA256."""
        if not settings.api_key_sha256:
            raise HTTPException(503, "Admin actions are disabled: API_KEY_SHA256 is not set")
        given = hashlib.sha256((x_api_key or "").encode()).hexdigest()
        if x_api_key is None or not hmac.compare_digest(given, settings.api_key_sha256):
            raise HTTPException(401, "Missing or wrong X-API-Key header")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        """Request ID, structured access log, request metrics, and a JSON 500 for crashes."""
        request_id = request.headers.get("x-request-id") or uuid4().hex
        start = time.perf_counter()
        IN_PROGRESS.inc()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("Unhandled error", extra={"request_id": request_id})
            response = JSONResponse(
                {"detail": "Internal error", "request_id": request_id}, status_code=500
            )
        finally:
            IN_PROGRESS.dec()
        seconds = time.perf_counter() - start
        endpoint = getattr(request.scope.get("route"), "path", "unmatched")
        REQUESTS.labels(endpoint, request.method, str(response.status_code)).inc()
        REQUEST_SECONDS.labels(endpoint).observe(seconds)
        log.log(
            logging.DEBUG if endpoint in QUIET else logging.INFO,
            "%s %s -> %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round(1000 * seconds, 1),
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # Failures of the services behind the gateway, as friendly HTTP answers.
    @app.exception_handler(ModelServerUnavailableError)
    def model_down(request: Request, exc: ModelServerUnavailableError) -> JSONResponse:
        log.warning("Model server unavailable: %s", exc)
        detail = "The recommender is starting or briefly unavailable. Try again in a few seconds."
        return JSONResponse({"detail": detail}, status_code=503)

    @app.exception_handler(QueryRejectedError)
    def rejected(request: Request, exc: QueryRejectedError) -> JSONResponse:
        if exc.unknown_track:
            detail = (
                "That track isn't in the current model's catalogue; find one with /tracks/search"
            )
            return JSONResponse({"detail": detail}, status_code=404)
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(RegistryError)
    def registry_conflict(request: Request, exc: RegistryError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(MlflowException)
    def registry_down(request: Request, exc: MlflowException) -> JSONResponse:
        log.warning("MLflow error: %s", exc)
        return JSONResponse({"detail": "The model registry is unavailable"}, status_code=503)

    @app.exception_handler(psycopg.Error)
    @app.exception_handler(psycopg_pool.PoolTimeout)
    def store_down(request: Request, exc: Exception) -> JSONResponse:
        log.warning("Feedback store error: %s", exc)
        return JSONResponse({"detail": "The feedback store is unavailable"}, status_code=503)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.post("/recommend", response_model=RecommendResponse, tags=["listener"])
    def recommend(body: RecommendRequest) -> RecommendResponse:
        """Top-k tracks for a seed track, slider values, or both."""
        sliders = body.sliders.model_dump(exclude_none=True)
        record = {**sliders, **({"track_id": body.track_id} if body.track_id else {})}
        rows = model.recommend(record, body.k)
        run_id = rows[0]["model_run_id"] if rows else ""
        version = None
        try:
            version = registry.version_of_run(run_id) if run_id else None
        except Exception:  # still answer; the run ID alone identifies the model
            log.warning("Couldn't map run %s to a model version", run_id, exc_info=True)
        request_id = uuid4()
        track_ids = [str(r["track_id"]) for r in rows]
        store.save_recommendation(
            request_id, body.track_id, sliders, body.k, version, run_id, track_ids
        )
        RECOMMENDATIONS.labels(str(version)).inc()
        return RecommendResponse(
            request_id=request_id,
            model_version=version,
            model_run_id=run_id,
            recommendations=[
                Recommendation(**_track(r), rank=int(r["rank"]), score=float(r["score"]))
                for r in rows
            ],
        )

    @app.post("/feedback", response_model=FeedbackResponse, status_code=201, tags=["listener"])
    def feedback(body: FeedbackRequest) -> FeedbackResponse:
        """A like or skip on one track of a list from /recommend."""
        shown = store.get_recommendation(body.request_id)
        if shown is None:
            raise HTTPException(404, "Unknown request_id: feedback must refer to a /recommend list")
        if body.track_id not in shown["track_ids"]:
            raise HTTPException(400, "That track wasn't in this recommendation list")
        rank = shown["track_ids"].index(body.track_id) + 1
        store.save_feedback(
            body.request_id,
            body.track_id,
            rank,
            body.action,
            shown["model_version"],
            shown["model_run_id"],
        )
        FEEDBACK.labels(body.action, str(shown["model_version"])).inc()
        return FeedbackResponse(model_version=shown["model_version"])

    @app.get("/tracks/search", response_model=SearchResponse, tags=["listener"])
    def search(
        q: str = Query(min_length=2, max_length=100, description="Part of a title or artist"),
        limit: int = Query(10, ge=1, le=50),
    ) -> SearchResponse:
        """Seed tracks the served model knows, best title matches first."""
        found = registry.search(q, limit)
        if found is None:
            raise HTTPException(503, "No model is being served yet, so there's nothing to search")
        version, hits = found
        tracks = [Track(**_track(row)) for row in hits.to_dict("records")]
        return SearchResponse(query=q, model_version=version, tracks=tracks)

    @app.get("/models", response_model=ModelsResponse, tags=["operator"])
    def models() -> ModelsResponse:
        """Registered versions, their aliases, lineage and like rate."""
        aliases = registry.aliases()
        try:
            feedback_counts = store.feedback_by_version()
        except Exception:  # the registry view is still useful without feedback
            log.warning("Couldn't read feedback counts", exc_info=True)
            feedback_counts = {}
        versions = []
        for row in registry.versions():
            counts = feedback_counts.get(row["version"], {"likes": 0, "skips": 0})
            total = counts["likes"] + counts["skips"]
            like_rate = round(counts["likes"] / total, 4) if total else None
            versions.append(ModelVersion(**row, **counts, like_rate=like_rate))
        return ModelsResponse(
            name=settings.model_name,
            champion=aliases.get("champion"),
            previous=aliases.get("previous"),
            challenger=aliases.get("challenger"),
            versions=versions,
        )

    @app.post(
        "/models/rollback",
        response_model=RollbackResponse,
        tags=["operator"],
        dependencies=[Depends(require_api_key)],
    )
    def rollback(body: RollbackRequest | None = None) -> RollbackResponse:
        """Make an earlier version champion (default: `previous`)."""
        champion, previous = registry.rollback(body.version if body else None)
        return RollbackResponse(
            champion=champion,
            previous=previous,
            detail=(
                f"Version {champion} is now the champion; the model server switches to it "
                "within a few seconds. Each response's model_version shows which one answered."
            ),
        )

    @app.get("/pipeline/runs", response_model=list[PipelineRun], tags=["operator"])
    def pipeline_runs(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
        """Recent pipeline runs, newest first."""
        return store.pipeline_runs(limit)

    @app.get("/pipeline/status", response_model=PipelineStatus, tags=["operator"])
    def pipeline_status() -> PipelineStatus:
        runs = store.pipeline_runs(1)
        return PipelineStatus(latest=runs[0] if runs else None, counts=store.pipeline_counts())

    @app.post(
        "/pipeline/trigger",
        response_model=PipelineRun,
        status_code=202,
        tags=["operator"],
        dependencies=[Depends(require_api_key)],
    )
    def trigger(body: TriggerRequest | None = None) -> dict:
        """Queue a pipeline run; the worker picks it up."""
        return store.request_pipeline_run((body or TriggerRequest()).trigger)

    @app.get("/drift", tags=["operator"])
    def drift() -> None:
        raise HTTPException(501, "Drift monitoring is not available yet")

    @app.get("/health", response_model=Health, tags=["service"])
    def health(response: Response) -> Health:
        """200 only when recommendations can be served: Postgres and the model server answer."""
        checks = {
            "postgres": "ok" if store.ping() else "unreachable",
            "model_server": "ok" if model.ping() else "unreachable",
            # Informational: serving continues while MLflow is down.
            "registry": (
                f"champion is version {registry.champion_version}"
                if registry.champion_version is not None
                else "no champion seen yet"
            ),
        }
        ok = checks["postgres"] == "ok" and checks["model_server"] == "ok"
        response.status_code = 200 if ok else 503
        return Health(status="ok" if ok else "unavailable", checks=checks)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
