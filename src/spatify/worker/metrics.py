"""The worker's Prometheus metrics (on METRICS_PORT, 9102): pipeline runs and data drift.

Drift scores as metrics are the problem statement's FR4; Phase 6 alerts on them.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, disable_created_metrics

disable_created_metrics()

RUNS = Counter(
    "spatify_pipeline_runs_total", "Pipeline runs, by trigger and result", ["trigger", "status"]
)
RUN_SECONDS = Histogram(
    "spatify_pipeline_run_seconds",
    "Wall-clock time of a pipeline run, by trigger",
    ["trigger"],
    buckets=(10, 20, 30, 45, 60, 90, 120, 180, 300, 600),
)
STAGE_SECONDS = Histogram(
    "spatify_pipeline_stage_seconds",
    "Wall-clock time of each stage that ran",
    ["stage"],
    buckets=(1, 2, 5, 10, 15, 20, 30, 60, 120),
)
QUEUE = Gauge("spatify_pipeline_queue_length", "Runs requested and not yet started")
LAST_RUN_OK = Gauge("spatify_pipeline_last_run_success", "1 if the last run succeeded, else 0")
LAST_RUN_TIME = Gauge("spatify_pipeline_last_run_timestamp_seconds", "When the last run finished")
QUARANTINED = Counter(
    "spatify_batches_quarantined_total", "Batch files moved to data/quarantine by validation"
)
DRIFT_PSI = Gauge(
    "spatify_drift_psi",
    "PSI of each feature in the latest batch, against the training data",
    ["feature"],
)
DRIFT_DETECTED = Gauge("spatify_drift_detected", "1 if the latest batch has drifted, else 0")
FEEDBACK_PENDING = Gauge(
    "spatify_feedback_pending", "Likes and skips since the last feedback snapshot (D37)"
)
RETRAIN_TO_REDEPLOY = Histogram(
    "spatify_retrain_to_redeploy_seconds",
    "From a retraining run's start to the model server answering with the promoted model",
    buckets=(30, 45, 60, 90, 120, 180, 240, 300, 600),
)
