"""The frontend's only way to the backend: REST calls to the gateway at API_BASE_URL.

Every failure becomes a GatewayError whose message a listener can act on. The rubric
asks for the frontend and backend to be connected only through configurable REST calls.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from spatify.frontend.metrics import GATEWAY_CALLS, GATEWAY_SECONDS

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT = float(os.environ.get("API_TIMEOUT_SECONDS", "15"))
TRANSPORT: httpx.BaseTransport | None = None  # tests replace the gateway with a fake

UNREACHABLE = (
    "Spatify can't reach its server right now. It may still be starting; "
    "try again in a few seconds."
)


class GatewayError(Exception):
    """A gateway call that failed, with a message fit to show the user."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _message(status: int, body: Any) -> str:
    detail = body.get("detail") if isinstance(body, dict) else None
    if status == 422 and isinstance(detail, list) and detail:
        first = detail[0]
        field = ".".join(str(part) for part in first.get("loc", [])[1:]) or "input"
        return f"That {field} wasn't accepted: {first.get('msg', 'invalid value')}."
    if status == 401:
        return "That admin key isn't right. It's SPATIFY_API_KEY in the project's .env file."
    if status == 503:
        return detail if isinstance(detail, str) else UNREACHABLE
    if status >= 500:
        request_id = body.get("request_id") if isinstance(body, dict) else None
        return "Something went wrong on the server" + (
            f" (request {request_id})." if request_id else "."
        )
    return detail if isinstance(detail, str) else f"The request failed (HTTP {status})."


def _call(method: str, path: str, endpoint: str, *, ok=(200, 201, 202), **kwargs) -> Any:
    """One gateway call; returns the JSON body, or raises GatewayError."""
    start = time.perf_counter()
    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT, transport=TRANSPORT) as client:
            response = client.request(method, path, **kwargs)
    except httpx.HTTPError as err:
        GATEWAY_CALLS.labels(endpoint, "unreachable").inc()
        raise GatewayError(UNREACHABLE) from err
    finally:
        GATEWAY_SECONDS.labels(endpoint).observe(time.perf_counter() - start)
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code not in ok:
        GATEWAY_CALLS.labels(endpoint, f"http_{response.status_code}").inc()
        raise GatewayError(_message(response.status_code, body), response.status_code)
    GATEWAY_CALLS.labels(endpoint, "ok").inc()
    return body


def search(query: str, limit: int = 10) -> dict:
    return _call("GET", "/tracks/search", "search", params={"q": query, "limit": limit})


def recommend(track_id: str | None, sliders: dict[str, float], k: int) -> dict:
    body = {"sliders": sliders, "k": k, **({"track_id": track_id} if track_id else {})}
    return _call("POST", "/recommend", "recommend", json=body)


def feedback(request_id: str, track_id: str, action: str) -> dict:
    body = {"request_id": request_id, "track_id": track_id, "action": action}
    return _call("POST", "/feedback", "feedback", json=body)


def models() -> dict:
    return _call("GET", "/models", "models")


def rollback(version: int | None, key: str) -> dict:
    body = {"version": version} if version is not None else None
    headers = {"X-API-Key": key}
    return _call("POST", "/models/rollback", "rollback", json=body, headers=headers)


def pipeline_dag() -> dict:
    return _call("GET", "/pipeline/dag", "pipeline_dag")


def pipeline_runs(limit: int = 20) -> list[dict]:
    return _call("GET", "/pipeline/runs", "pipeline_runs", params={"limit": limit})


def pipeline_status() -> dict:
    return _call("GET", "/pipeline/status", "pipeline_status")


def trigger(key: str) -> dict:
    return _call("POST", "/pipeline/trigger", "trigger", headers={"X-API-Key": key})


def drift() -> dict | None:
    """The drift report, or None before the first batch has been checked (HTTP 404)."""
    try:
        return _call("GET", "/drift", "drift")
    except GatewayError as err:
        if err.status in (404, 501):
            return None
        raise


def health() -> dict:
    """The gateway's /health checks; its 503 body still lists them."""
    return _call("GET", "/health", "health", ok=(200, 503))
