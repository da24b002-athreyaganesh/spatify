"""Calls to the model server (`mlflow models serve`, the MLflow scoring protocol)."""

from __future__ import annotations

import time

import httpx

from spatify.gateway.metrics import MODEL_ERRORS, MODEL_SECONDS


class ModelServerUnavailableError(Exception):
    """The model server can't be reached, timed out or failed (the gateway answers 503)."""


class QueryRejectedError(Exception):
    """The model server refused the query, e.g. an unknown seed track (the gateway answers 4xx)."""

    def __init__(self, message: str, unknown_track: bool) -> None:
        super().__init__(message)
        self.unknown_track = unknown_track


class ModelClient:
    def __init__(
        self, base_url: str, timeout: float, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def recommend(self, record: dict, k: int) -> list[dict]:
        """Top-k tracks for one query (a seed track_id and/or slider values)."""
        start = time.perf_counter()
        try:
            response = self.http.post(
                "/invocations", json={"dataframe_records": [record], "params": {"k": k}}
            )
        except httpx.HTTPError as err:
            MODEL_ERRORS.labels("unreachable").inc()
            raise ModelServerUnavailableError(f"model server unreachable: {err}") from err
        finally:
            MODEL_SECONDS.observe(time.perf_counter() - start)
        if response.status_code == 400:
            MODEL_ERRORS.labels("rejected").inc()
            message = response.json().get("message", "")
            # MLflow wraps our error as "Failed to predict data ... Error: <ours>".
            message = message.rsplit("Error: ", 1)[-1].strip()
            raise QueryRejectedError(message, unknown_track="Unknown seed track_id" in message)
        if response.status_code != 200:
            MODEL_ERRORS.labels(f"http_{response.status_code}").inc()
            raise ModelServerUnavailableError(f"model server answered HTTP {response.status_code}")
        return response.json()["predictions"]

    def ping(self) -> bool:
        try:
            return self.http.get("/ping").status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self.http.close()
