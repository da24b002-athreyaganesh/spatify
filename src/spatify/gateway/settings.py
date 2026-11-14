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
            log_level=env("LOG_LEVEL", cls.log_level),
        )
