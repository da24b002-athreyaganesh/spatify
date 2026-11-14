"""Prometheus metrics exposed at /metrics (L9: Counter, Gauge, Histogram, Info, with labels).

Phase 6 adds dashboards and alert rules on top of these.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, disable_created_metrics

disable_created_metrics()  # no *_created series next to every counter and histogram

# Buckets in seconds; the 0.3 s edge matches the p95 latency alert planned for Phase 6.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0)

REQUESTS = Counter(
    "spatify_gateway_requests_total",
    "HTTP requests handled by the gateway",
    ["endpoint", "method", "status"],
)
REQUEST_SECONDS = Histogram(
    "spatify_gateway_request_seconds",
    "Time to answer a request, by endpoint",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
)
IN_PROGRESS = Gauge("spatify_gateway_requests_in_progress", "Requests being handled now")

MODEL_SECONDS = Histogram(
    "spatify_model_server_request_seconds",
    "Time for the model server to answer /invocations",
    buckets=LATENCY_BUCKETS,
)
MODEL_ERRORS = Counter(
    "spatify_model_server_errors_total",
    "Failed calls to the model server, by kind (unreachable, rejected, http_<code>)",
    ["kind"],
)
MODEL_UP = Gauge(
    "spatify_model_server_up",
    "1 if the model server answered its last /ping (a blackbox probe), else 0",
)

RECOMMENDATIONS = Counter(
    "spatify_recommendations_total", "Recommendation lists served", ["model_version"]
)
FEEDBACK = Counter(
    "spatify_feedback_total", "Feedback events recorded", ["action", "model_version"]
)
DB_SECONDS = Histogram(
    "spatify_feedback_store_seconds",
    "Time for a feedback-store (Postgres) query, by operation",
    ["operation"],
    buckets=LATENCY_BUCKETS,
)
DB_ERRORS = Counter(
    "spatify_feedback_store_errors_total", "Failed feedback-store queries", ["operation"]
)
CHAMPION = Info("spatify_champion", "The registry's champion model, as last seen")
