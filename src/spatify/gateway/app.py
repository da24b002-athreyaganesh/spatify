"""The Spatify gateway: the REST API the frontend talks to (interactive docs at /docs).

Listener endpoints:  POST /recommend, POST /feedback, GET /tracks/search
Operator endpoints:  GET /models, POST /models/rollback*, GET /pipeline/status,
                     GET /pipeline/runs, GET /pipeline/dag, POST /pipeline/trigger*,
                     GET /drift, GET /alerts
Service endpoints:   GET /health (503 until Postgres and the model server answer), GET /metrics,
                     POST /alerts** (Alertmanager's webhook, D41)
* needs the admin key in the X-API-Key header; ** as a Bearer token.

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
from pathlib import Path
from uuid import uuid4

import psycopg
import psycopg_pool
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from mlflow.exceptions import MlflowException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from spatify.gateway.feedback_drift import feedback_drift
from spatify.gateway.logs import setup_logging
from spatify.gateway.metrics import (
    ALERTS,
    CHAMPION,
    EXPLORED,
    FEEDBACK,
    FEEDBACK_DRIFT,
    FEEDBACK_RANK,
    IN_PROGRESS,
    LIKE_RATE,
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
from spatify.gateway.pipeline_view import PipelineUnavailableError, drift_view, load_dag
from spatify.gateway.registry import Registry, RegistryError
from spatify.gateway.schemas import (
    AlertmanagerWebhook,
    AlertRecord,
    DriftReport,
    FeedbackRequest,
    FeedbackResponse,
    Health,
    ModelsResponse,
    ModelVersion,
    PipelineDag,
    PipelineRun,
    PipelineStatus,
    Problem,
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

STORE_DOWN = "The feedback store (Postgres) is unavailable"
NO_KEY = "Admin actions are disabled: API_KEY_SHA256 isn't set"


def errors(codes: dict[int, str]) -> dict[int, dict]:
    """OpenAPI entries for the errors a route can answer with (shown at /docs, the LLD)."""
    return {code: {"model": Problem, "description": text} for code, text in codes.items()}


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

    def like_rate_report() -> dict:
        stats = store.like_rates(settings.feedback_recent, settings.feedback_baseline)
        return feedback_drift(
            stats, settings.feedback_recent, settings.feedback_baseline, settings.feedback_drop
        )

    def watch() -> None:
        """Every refresh_seconds: probe the model server, re-read the champion, and update
        the like-rate gauges."""
        while True:
            MODEL_UP.set(1 if model.ping() else 0)
            try:
                registry.refresh()
                if registry.champion_version is not None:
                    CHAMPION.info({"version": str(registry.champion_version)})
            except Exception:
                log.warning("Couldn't refresh the champion from MLflow", exc_info=True)
            try:
                report = like_rate_report()
                for window in ("recent", "baseline"):
                    if report[f"{window}_like_rate"] is not None:
                        LIKE_RATE.labels(window).set(report[f"{window}_like_rate"])
                FEEDBACK_DRIFT.set(1 if report["drifted"] else 0)
            except Exception:
                log.warning("Couldn't compute the like rates", exc_info=True)
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

    def key_ok(key: str | None) -> bool:
        """True for the key whose SHA-256 is in API_KEY_SHA256 (D28)."""
        if not settings.api_key_sha256:
            raise HTTPException(503, NO_KEY)
        given = hashlib.sha256((key or "").encode()).hexdigest()
        return key is not None and hmac.compare_digest(given, settings.api_key_sha256)

    def require_api_key(x_api_key: str | None = Header(None)) -> None:
        """Admin actions need the key in the X-API-Key header."""
        if not key_ok(x_api_key):
            raise HTTPException(401, "Missing or wrong X-API-Key header")

    def require_bearer_key(authorization: str | None = Header(None)) -> None:
        """Alertmanager sends the same key as `Authorization: Bearer <key>` (D41)."""
        scheme, _, key = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not key_ok(key.strip() or None):
            raise HTTPException(401, "Missing or wrong Bearer token")

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

    @app.exception_handler(PipelineUnavailableError)
    def pipeline_missing(request: Request, exc: PipelineUnavailableError) -> JSONResponse:
        log.warning("Pipeline view unavailable: %s", exc)
        detail = "The pipeline definition isn't available to the gateway (PIPELINE_DIR)"
        return JSONResponse({"detail": detail}, status_code=503)

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

    @app.post(
        "/recommend",
        response_model=RecommendResponse,
        tags=["listener"],
        responses=errors({
            400: "The query can't be answered: no seed and no slider this model uses",
            404: "The seed track isn't in the served model's catalogue (see /tracks/search)",
            503: "The model server is starting or switching versions (try again in a few "
                 "seconds), or the feedback store is unavailable",
        }),
    )  # fmt: skip
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
        # Exploration (D37): random tracks take the lowest-ranked slots, so feedback also
        # covers songs the model wouldn't have picked, and the next model learns from them.
        picks = registry.explore(
            int(settings.exploration_rate * len(rows)),
            exclude={str(r["track_id"]) for r in rows} | {body.track_id},
        )
        explored = [p["track_id"] for p in picks]
        scores = registry.similarity(record, explored, run_id)
        slots = range(len(rows) - len(picks), len(rows))
        for pick, slot, score_ in zip(picks, slots, scores, strict=True):
            rows[slot] = {**pick, "rank": rows[slot]["rank"], "score": score_}
        EXPLORED.inc(len(picks))

        request_id = uuid4()
        track_ids = [str(r["track_id"]) for r in rows]
        store.save_recommendation(
            request_id, body.track_id, sliders, body.k, version, run_id, track_ids, explored
        )
        RECOMMENDATIONS.labels(str(version)).inc()
        why = registry.why(record, track_ids, run_id)
        return RecommendResponse(
            request_id=request_id,
            model_version=version,
            model_run_id=run_id,
            recommendations=[
                Recommendation(
                    **_track(r),
                    rank=int(r["rank"]),
                    score=float(r["score"]),
                    why=[] if r["track_id"] in explored else labels,
                    explored=r["track_id"] in explored,
                )
                for r, labels in zip(rows, why, strict=True)
            ],
        )

    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        status_code=201,
        tags=["listener"],
        responses=errors({
            400: "That track wasn't in this recommendation list",
            404: "Unknown request_id: feedback must refer to a list from /recommend",
            503: STORE_DOWN,
        }),
    )  # fmt: skip
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
        FEEDBACK_RANK.labels(body.action, str(shown["model_version"])).observe(rank)
        return FeedbackResponse(model_version=shown["model_version"])

    @app.get(
        "/tracks/search",
        response_model=SearchResponse,
        tags=["listener"],
        responses=errors({503: "No model is being served yet, so there's nothing to search"}),
    )
    def search(
        q: str = Query(min_length=2, max_length=100, description="Part of a title or artist"),
        limit: int = Query(10, ge=1, le=50, description="Most tracks to return"),
    ) -> SearchResponse:
        """Seed tracks the served model knows, best title matches first."""
        found = registry.search(q, limit)
        if found is None:
            raise HTTPException(503, "No model is being served yet, so there's nothing to search")
        version, hits = found
        tracks = [Track(**_track(row)) for row in hits.to_dict("records")]
        return SearchResponse(query=q, model_version=version, tracks=tracks)

    @app.get(
        "/models",
        response_model=ModelsResponse,
        tags=["operator"],
        responses=errors({503: "The model registry (MLflow) is unavailable"}),
    )
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
        responses=errors({
            401: "Missing or wrong X-API-Key header",
            409: "Impossible rollback: no champion, no `previous`, the version is already "
                 "the champion, or it doesn't exist",
            503: "The model registry is unavailable, or admin actions are disabled",
        }),
    )  # fmt: skip
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

    @app.post(
        "/alerts",
        status_code=204,
        dependencies=[Depends(require_bearer_key)],
        tags=["service"],
        responses=errors(
            {
                401: "Missing or wrong Bearer token",
                503: f"{STORE_DOWN}, or admin actions are disabled",
            }
        ),
    )
    def receive_alerts(body: AlertmanagerWebhook) -> Response:
        """Alertmanager's webhook: record each alert, firing or resolved (D41)."""
        rows = []
        for alert in body.alerts:
            ends = alert.endsAt if alert.endsAt and alert.endsAt.year > 2000 else None
            rows.append(
                {
                    "fingerprint": alert.fingerprint,
                    "starts_at": alert.startsAt,
                    "ends_at": ends if alert.status == "resolved" else None,
                    "status": alert.status,
                    "alertname": alert.labels.get("alertname", "unnamed"),
                    "severity": alert.labels.get("severity"),
                    "summary": alert.annotations.get("summary"),
                    "description": alert.annotations.get("description"),
                    "labels": alert.labels,
                }
            )
            ALERTS.labels(rows[-1]["alertname"], alert.status).inc()
            log.warning(
                "Alert %s %s: %s",
                rows[-1]["alertname"],
                alert.status,
                rows[-1]["summary"] or "",
                extra={"alert": alert.labels, "status": alert.status},
            )
        store.save_alerts(rows)
        return Response(status_code=204)

    @app.get(
        "/alerts",
        response_model=list[AlertRecord],
        tags=["operator"],
        responses=errors({503: STORE_DOWN}),
    )
    def alerts(
        limit: int = Query(50, ge=1, le=500, description="Most alerts to return"),
    ) -> list[dict]:
        """Alerts from Prometheus (through Alertmanager): firing first, then the latest."""
        return store.alerts(limit)

    @app.get(
        "/pipeline/runs",
        response_model=list[PipelineRun],
        tags=["operator"],
        responses=errors({503: STORE_DOWN}),
    )
    def pipeline_runs(
        limit: int = Query(20, ge=1, le=200, description="Most runs to return"),
    ) -> list[dict]:
        """Recent pipeline runs, newest first."""
        return store.pipeline_runs(limit)

    @app.get(
        "/pipeline/dag",
        response_model=PipelineDag,
        tags=["operator"],
        responses=errors({503: "The pipeline files (dvc.yaml, dvc.lock) aren't mounted"}),
    )
    def pipeline_dag() -> dict:
        """The DVC pipeline: stages, data flow, each stage's last result and throughput."""
        try:
            runs = store.pipeline_runs(1)
        except Exception:  # the diagram is still useful without the run history
            log.warning("Couldn't read the latest pipeline run", exc_info=True)
            runs = []
        return load_dag(Path(settings.pipeline_dir), runs[0] if runs else None)

    @app.get(
        "/pipeline/status",
        response_model=PipelineStatus,
        tags=["operator"],
        responses=errors({503: STORE_DOWN}),
    )
    def pipeline_status() -> PipelineStatus:
        runs = store.pipeline_runs(1)
        return PipelineStatus(latest=runs[0] if runs else None, counts=store.pipeline_counts())

    @app.post(
        "/pipeline/trigger",
        response_model=PipelineRun,
        status_code=202,
        tags=["operator"],
        dependencies=[Depends(require_api_key)],
        responses=errors(
            {
                401: "Missing or wrong X-API-Key header",
                503: f"{STORE_DOWN}, or admin actions are disabled",
            }
        ),
    )
    def trigger(body: TriggerRequest | None = None) -> dict:
        """Queue a pipeline run; the worker picks it up."""
        return store.request_pipeline_run((body or TriggerRequest()).trigger)

    @app.get(
        "/drift",
        response_model=DriftReport,
        tags=["operator"],
        responses=errors({404: "No drift report yet: no batch checked and no feedback"}),
    )
    def drift() -> dict:
        """Each incoming batch's drift from the training data (PSI and KS), newest first,
        and feedback drift: the like rate of the latest likes and skips against before."""
        try:
            seen = store.incoming_batches()
        except Exception:  # the report alone is still useful
            log.warning("Couldn't read the batch records", exc_info=True)
            seen = []
        view = drift_view(Path(settings.pipeline_dir), seen)
        try:
            likes = like_rate_report()
        except Exception:
            log.warning("Couldn't compute the like rates", exc_info=True)
            likes = None
        if view is None and not (likes and likes["recent_events"]):
            raise HTTPException(
                404, "No drift report yet: the worker checks each batch dropped into data/incoming"
            )
        view = view or {"reference_rows": None, "thresholds": {}, "batches": []}
        return {**view, "feedback": likes}

    @app.get(
        "/health",
        response_model=Health,
        tags=["service"],
        responses={503: {"model": Health, "description": "Postgres or the model server is down"}},
    )
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
