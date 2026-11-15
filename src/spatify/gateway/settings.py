"""Gateway settings, read from environment variables (set in docker-compose.yml and .env).

Every other service is reached by its Compose service name, never by an IP address.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_server_url: str = "http://model-server:8080"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    database_url: str = "postgresql://spatify@postgres:5432/spatify"
    model_name: str = "spatify"
    # SHA-256 of the admin API key. Rollback and pipeline triggers are refused while empty.
    api_key_sha256: str = ""
    model_timeout_seconds: float = 5.0
    # How often the champion version and the model server's health are re-checked.
    refresh_seconds: float = 15.0
    # Read-only mount of dvc.yaml, dvc.lock and metrics/, for the console's pipeline view.
    pipeline_dir: str = "/app/pipeline"
    # Share of each list given to random tracks, so feedback isn't only about what the
    # model already ranks high (D37). int(rate * k) slots: one in a list of 10.
    exploration_rate: float = 0.1
    # Feedback drift (D37): the latest likes and skips against the ones before them.
    feedback_recent: int = 200
    feedback_baseline: int = 1000
    feedback_drop: float = 0.15
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            model_server_url=env("MODEL_SERVER_URL", cls.model_server_url),
            mlflow_tracking_uri=env("MLFLOW_TRACKING_URI", cls.mlflow_tracking_uri),
            database_url=env("DATABASE_URL", cls.database_url),
            model_name=env("MODEL_NAME", cls.model_name),
            api_key_sha256=env("API_KEY_SHA256", cls.api_key_sha256).strip().lower(),
            model_timeout_seconds=float(env("MODEL_TIMEOUT_SECONDS", cls.model_timeout_seconds)),
            refresh_seconds=float(env("REFRESH_SECONDS", cls.refresh_seconds)),
            pipeline_dir=env("PIPELINE_DIR", cls.pipeline_dir),
            exploration_rate=float(env("EXPLORATION_RATE", cls.exploration_rate)),
            feedback_recent=int(env("FEEDBACK_RECENT", cls.feedback_recent)),
            feedback_baseline=int(env("FEEDBACK_BASELINE", cls.feedback_baseline)),
            feedback_drop=float(env("FEEDBACK_DROP", cls.feedback_drop)),
            log_level=env("LOG_LEVEL", cls.log_level),
        )
