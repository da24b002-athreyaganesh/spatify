"""Prometheus metrics exposed at /metrics (L9: Counter, Gauge, Histogram, Summary, Info).

The dashboards and alert rules in monitoring/ are built on these (D40, D41).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, Summary, disable_created_metrics

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

EXPLORED = Counter("spatify_explored_slots_total", "List slots given to random tracks (D37)")
LIKE_RATE = Gauge(
    "spatify_feedback_like_rate",
    "Like rate of the latest likes and skips (recent) and of the ones before them (baseline)",
    ["window"],
)
FEEDBACK_DRIFT = Gauge(
    "spatify_feedback_drift", "1 if the like rate has dropped by the threshold or more, else 0"
)
FEEDBACK_RANK = Summary(
    "spatify_feedback_rank",
    "Position in its list (1 = top) of each liked or skipped track: liked songs near the "
    "top mean the model's order matches listeners' taste",
    ["action", "model_version"],
)
ALERTS = Counter(
    "spatify_alert_notifications_total",
    "Alert notices received from Alertmanager (D41)",
    ["alertname", "status"],
)
