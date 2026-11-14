"""The feedback store (Postgres): served lists, feedback events and pipeline runs.

Every list the gateway serves is saved with the model version that produced it, and
feedback must name one of those lists. So each like or skip is tied to the exact model
and position it was shown at (FR2), which Phase 5 needs for hit-rate and like rate per
version. The gateway creates the tables at start-up if they don't exist.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from spatify.db_schema import SCHEMA
from spatify.gateway.metrics import DB_ERRORS, DB_SECONDS


class Store:
    def __init__(self, url: str) -> None:
        self.pool = ConnectionPool(
            url, min_size=1, max_size=5, open=False, kwargs={"row_factory": dict_row}
        )

    def open(self, timeout: float = 30.0) -> None:
        """Connect and create the tables; raises if Postgres stays unreachable."""
        self.pool.open(wait=True, timeout=timeout)
        with self._timed("init_schema") as conn:
            conn.execute(SCHEMA)

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def _timed(self, operation: str) -> Iterator:
        start = time.perf_counter()
        try:
            with self.pool.connection(timeout=5) as conn:
                yield conn
        except Exception:
            DB_ERRORS.labels(operation).inc()
            raise
        finally:
            DB_SECONDS.labels(operation).observe(time.perf_counter() - start)

    def ping(self) -> bool:
        try:
            with self._timed("ping") as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def save_recommendation(
        self,
        request_id: UUID,
        seed_track_id: str | None,
        sliders: dict,
        k: int,
        model_version: int | None,
        model_run_id: str,
        track_ids: list[str],
    ) -> None:
        with self._timed("save_recommendation") as conn:
            conn.execute(
                "INSERT INTO recommendations (request_id, seed_track_id, sliders, k,"
                " model_version, model_run_id, track_ids) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (request_id, seed_track_id, Jsonb(sliders), k, model_version, model_run_id,
                 track_ids),
            )  # fmt: skip

    def get_recommendation(self, request_id: UUID) -> dict | None:
        with self._timed("get_recommendation") as conn:
            return conn.execute(
                "SELECT track_ids, model_version, model_run_id FROM recommendations"
                " WHERE request_id = %s",
                (request_id,),
            ).fetchone()

    def save_feedback(
        self,
        request_id: UUID,
        track_id: str,
        rank: int,
        action: str,
        model_version: int | None,
        model_run_id: str,
    ) -> None:
        """Record a like or skip; a second click on the same track replaces the first."""
        with self._timed("save_feedback") as conn:
            conn.execute(
                "INSERT INTO feedback (request_id, track_id, rank, action, model_version,"
                " model_run_id) VALUES (%s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (request_id, track_id)"
                " DO UPDATE SET action = EXCLUDED.action, created_at = now()",
                (request_id, track_id, rank, action, model_version, model_run_id),
            )

    def feedback_by_version(self) -> dict[int, dict[str, int]]:
        """version -> {"likes": n, "skips": n}."""
        with self._timed("feedback_by_version") as conn:
            rows = conn.execute(
                "SELECT model_version,"
                " count(*) FILTER (WHERE action = 'like') AS likes,"
                " count(*) FILTER (WHERE action = 'skip') AS skips"
                " FROM feedback WHERE model_version IS NOT NULL GROUP BY model_version"
            ).fetchall()
        return {r["model_version"]: {"likes": r["likes"], "skips": r["skips"]} for r in rows}

    def pipeline_runs(self, limit: int) -> list[dict]:
        with self._timed("pipeline_runs") as conn:
            return conn.execute(
                "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()

    def pipeline_counts(self) -> dict[str, int]:
        with self._timed("pipeline_counts") as conn:
            rows = conn.execute(
                "SELECT status, count(*) AS n FROM pipeline_runs GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def incoming_batches(self) -> list[dict]:
        """Batch files the worker has seen, newest first."""
        with self._timed("incoming_batches") as conn:
            return conn.execute(
                "SELECT name, status, first_seen, drifted, message FROM incoming_batches"
                " ORDER BY first_seen DESC, name"
            ).fetchall()

    def request_pipeline_run(self, trigger: str) -> dict:
        """Queue a pipeline run; the worker (Phase 5) picks up `requested` rows."""
        with self._timed("request_pipeline_run") as conn:
            return conn.execute(
                "INSERT INTO pipeline_runs (trigger) VALUES (%s) RETURNING *", (trigger,)
            ).fetchone()
