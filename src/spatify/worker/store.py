"""The worker's records in Postgres: pipeline runs and the batch files it has seen."""

from __future__ import annotations

from psycopg import connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from spatify.db_schema import SCHEMA

RUN_FIELDS = {"status", "trigger", "stages", "message", "model_version", "promoted"}


class WorkerStore:
    """A short connection per call: the worker is single-threaded and calls rarely."""

    def __init__(self, url: str) -> None:
        self.url = url

    def _run(self, sql: str, params: tuple = (), fetch: str | None = None):
        with connect(self.url, autocommit=True, row_factory=dict_row) as conn:
            cursor = conn.execute(sql, params)
            if fetch == "one":
                return cursor.fetchone()
            if fetch == "all":
                return cursor.fetchall()
        return None

    def init_schema(self) -> None:
        self._run(SCHEMA)

    def fail_interrupted_runs(self) -> int:
        """Runs a previous worker left 'running' (it stopped mid-run) become 'failed'."""
        rows = self._run(
            "UPDATE pipeline_runs SET status = 'failed', finished_at = now(),"
            " message = 'The worker stopped during this run.'"
            " WHERE status = 'running' RETURNING id",
            fetch="all",
        )
        return len(rows)

    def retry_failed_batches(self) -> int:
        """Batches that failed go back in line: a restart is how the operator says "retry"."""
        rows = self._run(
            "UPDATE incoming_batches SET status = 'new', message = NULL"
            " WHERE status = 'failed' RETURNING name",
            fetch="all",
        )
        return len(rows)

    def claim_requested(self) -> dict | None:
        """Take the oldest queued run; SKIP LOCKED keeps two workers off the same run."""
        return self._run(
            "UPDATE pipeline_runs SET status = 'running', started_at = now()"
            " WHERE id = (SELECT id FROM pipeline_runs WHERE status = 'requested'"
            " ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING *",
            fetch="one",
        )

    def start_run(self, trigger: str) -> dict:
        return self._run(
            "INSERT INTO pipeline_runs (trigger, status, started_at)"
            " VALUES (%s, 'running', now()) RETURNING *",
            (trigger,),
            fetch="one",
        )

    def update_run(self, run_id: int, finished: bool = False, **fields) -> None:
        unknown = set(fields) - RUN_FIELDS
        if unknown:
            raise ValueError(f"unknown run fields {unknown}")
        values = {k: Jsonb(v) if k == "stages" else v for k, v in fields.items()}
        sets = [f"{k} = %s" for k in values] + (["finished_at = now()"] if finished else [])
        self._run(
            f"UPDATE pipeline_runs SET {', '.join(sets)} WHERE id = %s",  # noqa: S608 - names are fixed
            (*values.values(), run_id),
        )

    def queue_length(self) -> int:
        row = self._run(
            "SELECT count(*) AS n FROM pipeline_runs WHERE status = 'requested'", fetch="one"
        )
        return row["n"]

    def batches(self) -> dict[str, dict]:
        """name -> {sha256, status} for every batch file seen so far."""
        rows = self._run("SELECT name, sha256, status FROM incoming_batches", fetch="all")
        return {r["name"]: r for r in rows}

    def record_batch(self, name: str, sha256: str, run_id: int) -> None:
        """A new batch file, or a known name with new contents: back to 'new'."""
        self._run(
            "INSERT INTO incoming_batches (name, sha256, run_id) VALUES (%s, %s, %s)"
            " ON CONFLICT (name) DO UPDATE SET sha256 = EXCLUDED.sha256, status = 'new',"
            " run_id = EXCLUDED.run_id, first_seen = now(), drifted = NULL, message = NULL",
            (name, sha256, run_id),
        )

    def set_batch(
        self,
        name: str,
        status: str,
        run_id: int,
        drifted: bool | None = None,
        message: str | None = None,
    ) -> None:
        self._run(
            "UPDATE incoming_batches SET status = %s, run_id = %s, drifted = %s, message = %s"
            " WHERE name = %s",
            (status, run_id, drifted, message, name),
        )
