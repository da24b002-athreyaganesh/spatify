"""Frontend metrics, served by prometheus_client on their own port (as in A4).

Streamlit serves the app on 8501 and has no /metrics of its own, so a small HTTP server in
a background thread exposes these on METRICS_PORT (9101), for Prometheus in Phase 6.
"""

from __future__ import annotations

import logging

from prometheus_client import (
    REGISTRY,
    Counter,
    Histogram,
    disable_created_metrics,
    start_http_server,
)

log = logging.getLogger(__name__)

NAMESPACE = "spatify_frontend"
disable_created_metrics()


def _metric(cls, name: str, documentation: str, labelnames=(), **kwargs):
    """Create a metric, or return it if it already exists (A4's pattern).

    Streamlit can re-import this module; registering the same metric twice in the
    process-wide REGISTRY would raise "Duplicated timeseries" and crash the app.
    """
    full = f"{NAMESPACE}_{name}"
    existing = REGISTRY._names_to_collectors.get(full) or REGISTRY._names_to_collectors.get(
        full.removesuffix("_total")
    )
    if existing is not None:
        return existing
    return cls(name, documentation, labelnames, namespace=NAMESPACE, **kwargs)


PAGE_VIEWS = _metric(Counter, "page_views_total", "Pages shown, by page", ["page"])
GATEWAY_CALLS = _metric(
    Counter,
    "gateway_calls_total",
    "Calls to the gateway, by endpoint and outcome (ok, http_<code>, unreachable)",
    ["endpoint", "outcome"],
)
GATEWAY_SECONDS = _metric(
    Histogram,
    "gateway_call_seconds",
    "Time for the gateway to answer, as the frontend sees it",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0, 10.0),
)
FEEDBACK_CLICKS = _metric(Counter, "feedback_clicks_total", "Like and skip clicks", ["action"])
ERRORS_SHOWN = _metric(
    Counter, "errors_shown_total", "Error messages shown to users, by kind", ["kind"]
)


def start_metrics_server(port: int) -> bool:
    """Serve /metrics on `port`; False if that fails (the app keeps working without it)."""
    try:
        start_http_server(port)
    except OSError:
        log.warning("Couldn't serve frontend metrics on port %d", port, exc_info=True)
        return False
    log.info("Frontend metrics on port %d", port)
    return True
