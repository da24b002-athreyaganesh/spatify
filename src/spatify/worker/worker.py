"""What the worker does on each check (D36).

1. A run requested from the console or the API (`pipeline_runs`, status 'requested'):
   bring the data up to date, then retrain (`dvc repro`; if training was up to date,
   force it, since someone asked for a retrain).
2. Otherwise, new, changed or removed batch files in data/incoming:
   `dvc add data/incoming`, then `dvc repro drift_check` (ingest, validate, drift check).
   Retrain only if a new batch has drifted (trigger "drift"); otherwise the data is
   recorded and the model left alone, as the problem statement's FR5 asks.
3. Otherwise, if `feedback.retrain_every` likes and skips have arrived since the last
   snapshot: retrain (trigger "feedback", D37).
4. Otherwise, nothing.

Every retrain first exports all feedback to data/feedback/feedback.parquet and runs
`dvc add data/feedback`, so each model is traced to the exact likes and skips it learned
from. After a promotion the worker waits until the model server answers with the new
model and records how long that took (retrain-to-redeploy time, D37).

A failed run leaves the champion serving: only the register stage can move the alias,
and it never runs after a failed stage. A batch that validation rejects is moved to
data/quarantine/, so it can't fail again on every check. Every run is recorded in
Postgres with its trigger, result, per-stage times, model version and a message.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from spatify.worker import metrics
from spatify.worker.runner import DvcResult, run_dvc

log = logging.getLogger("spatify.worker")

INCOMING = Path("data/incoming")
QUARANTINE = Path("data/quarantine")
DATA_STAGES = ("ingest", "validate", "features")  # their metrics files hold rows and rows/s
FEEDBACK = Path("data/feedback")
SNAPSHOT = FEEDBACK / "feedback.parquet"
SNAPSHOT_TYPES = {
    "request_id": "string", "created_at": "datetime64[us, UTC]", "seed_track_id": "string",
    "sliders": "string", "track_id": "string", "rank": "Int64", "action": "string",
    "explored": "boolean", "model_version": "Int64",
}  # fmt: skip


@dataclass
class Changes:
    new: dict[str, str] = field(default_factory=dict)  # name -> sha256: new, changed or pending
    removed: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.new or self.removed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Worker:
    def __init__(
        self,
        repo: Path,
        store,
        dvc: Callable[[list[str], Path, Mapping[str, str]], DvcResult] = run_dvc,
        env: Mapping[str, str] | None = None,
        served: Callable[[str], float | None] | None = None,
    ) -> None:
        self.repo, self.store, self.dvc = repo, store, dvc
        self.env = dict(os.environ if env is None else env)
        self.served = served or self.wait_until_served
        self.last_failed = False  # the main loop waits a full poll after a failed run
        self.record_throughput()  # the dashboards show the last numbers from the start

    def step(self) -> bool:
        """One check. True if it ran the pipeline."""
        metrics.QUEUE.set(self.store.queue_length())
        run = self.store.claim_requested()
        if run is not None:
            self.handle(run, self.incoming_changes(), retrain=True)
            return True
        changes = self.incoming_changes()
        if changes:
            self.handle(self.store.start_run("new_data"), changes, retrain=False)
            return True
        pending = self.store.feedback_count() - self.snapshot_rows()
        metrics.FEEDBACK_PENDING.set(pending)
        if pending >= self.param("feedback", "retrain_every", 2000):
            self.handle(self.store.start_run("feedback"), Changes(), retrain=True)
            return True
        return False

    def param(self, section: str, key: str, default):
        """A setting from the repo's params.yaml, read fresh so edits apply at once."""
        try:
            return yaml.safe_load((self.repo / "params.yaml").read_text())[section][key]
        except (OSError, KeyError, TypeError):
            return default

    def snapshot_rows(self) -> int:
        try:
            return pq.ParquetFile(self.repo / SNAPSHOT).metadata.num_rows
        except (OSError, ValueError):
            return 0

    def export_feedback(self) -> int:
        """Write every like and skip to the snapshot and version it with DVC."""
        events = pd.DataFrame(self.store.export_feedback(), columns=list(SNAPSHOT_TYPES))
        (self.repo / FEEDBACK).mkdir(exist_ok=True)
        events.astype(SNAPSHOT_TYPES).to_parquet(self.repo / SNAPSHOT, index=False)
        added = self.dvc(["add", str(FEEDBACK)], self.repo, self.env)
        if not added.ok:
            raise RuntimeError(f"dvc add data/feedback failed: {error_line(added.tail)}")
        return len(events)

    def wait_until_served(self, run_id: str, timeout: float = 180) -> float | None:
        """Seconds until the model server answers with the model from `run_id` (None: never)."""
        url = self.env.get("MODEL_SERVER_URL")
        if not url:
            return None
        body = json.dumps({"dataframe_records": [{"energy": 0.5}], "params": {"k": 1}}).encode()
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                request = urllib.request.Request(
                    f"{url}/invocations", body, {"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    answer = json.load(response)["predictions"][0]["model_run_id"]
                if answer == run_id:
                    return round(time.monotonic() - start, 1)
            except Exception:  # the server is restarting; keep asking
                pass
            time.sleep(1)
        return None

    def incoming_changes(self) -> Changes:
        """Batch files that are new, changed, still waiting or back after being removed,
        and ones that were removed."""
        known = self.store.batches()
        files = {p.name: sha256(p) for p in sorted((self.repo / INCOMING).glob("*.csv"))}
        changes = Changes()
        for name, digest in files.items():
            row = known.get(name)
            # "removed": dropped in again with the same contents, so it's new data again.
            if row is None or row["sha256"] != digest or row["status"] in ("new", "removed"):
                changes.new[name] = digest
        changes.removed = [
            name for name, row in known.items()
            if name not in files and row["status"] in ("new", "processed")
        ]  # fmt: skip
        return changes

    def handle(self, run: dict, changes: Changes, retrain: bool) -> None:
        trigger, run_id = run["trigger"], run["id"]
        start, stages, notes = time.monotonic(), {}, []
        status, version, promoted = "succeeded", None, None
        try:
            for name, digest in changes.new.items():
                self.store.record_batch(name, digest, run_id)
            for name in changes.removed:
                self.store.set_batch(name, "removed", run_id)
            if changes:
                added = self.dvc(["add", str(INCOMING)], self.repo, self.env)
                if not added.ok:
                    raise RuntimeError(f"dvc add failed: {error_line(added.tail)}")

            # The data stages and the drift check.
            data = self.repro(["repro", "drift_check"], stages, trigger)
            if not data.ok:
                status = "failed"
                notes.append(self.after_failed_data(run_id, changes, data))
                return
            drifted = self.record_drift(run_id, changes)

            # Retrain?
            if not (retrain or drifted):
                notes.append(
                    f"Checked {', '.join(changes.new)}: no drift, so the model wasn't retrained."
                    if changes.new
                    else "Brought the data stages up to date; no new batches."
                )
                return
            if drifted and not retrain:
                trigger = "drift"
                self.store.update_run(run_id, trigger=trigger)
                notes.append(f"Drift in {', '.join(drifted)}, so the model was retrained.")
            if trigger == "feedback":
                notes.append("Enough new likes and skips arrived, so the model was retrained.")
            exported = self.export_feedback()
            notes.append(f"Training saw {exported:,} likes and skips.")
            full = self.repro(["repro"], stages, trigger)
            if full.ok and retrain and stages.get("train", {}).get("status") != "ok":
                full = self.repro(["repro", "--force", "--downstream", "train"], stages, trigger)
            if not full.ok:
                status = "failed"
                notes.append(
                    f"Retraining failed at {failed_stage(full)}; the champion keeps serving."
                )
                return
            if stages.get("register", {}).get("status") == "ok":
                result = self.read_json("metrics/register.json") or {}
                version, promoted = result.get("version"), result.get("promoted")
                metric = result.get("metric", "genre_match_at_k")
                notes.append(
                    f"Registered version {version} ({result.get('variant')}): promoted on {metric}."
                    if promoted
                    else f"Registered version {version}: not promoted on {metric} "
                    f"(gain {result.get('gain')}); version {result.get('champion_version')} "
                    "keeps serving."
                )
                if promoted:
                    notes.append(self.record_redeploy(result, stages, start))
        except Exception as err:  # record any surprise as a failed run; keep the worker alive
            log.exception("Run %s failed", run_id)
            status = "failed"
            notes.append(f"The worker hit an error: {err}. {RETRY}")
            for name in changes.new:  # so the same error doesn't repeat on every check
                self.store.set_batch(name, "failed", run_id, message=str(err)[:200])
        finally:
            self.last_failed = status == "failed"
            seconds = time.monotonic() - start
            message = " ".join(notes) or None
            self.store.update_run(
                run_id, finished=True, status=status, stages=stages, message=message,
                model_version=version, promoted=promoted,
            )  # fmt: skip
            metrics.RUNS.labels(trigger, status).inc()
            metrics.RUN_SECONDS.labels(trigger).observe(seconds)
            metrics.LAST_RUN_OK.set(1 if status == "succeeded" else 0)
            metrics.LAST_RUN_TIME.set(time.time())
            self.record_throughput()
            log.info("Run %s (%s) %s in %.0f s. %s", run_id, trigger, status, seconds, message)

    def record_throughput(self) -> None:
        """Each data stage's rows and rows per second, from its DVC metrics file (D40)."""
        for stage in DATA_STAGES:
            report = self.read_json(f"metrics/{stage}.json") or {}
            if "rows_per_s" in report:
                metrics.STAGE_ROWS_PER_SECOND.labels(stage).set(report["rows_per_s"])
                metrics.STAGE_ROWS.labels(stage).set(report.get("rows", 0))

    def record_redeploy(self, result: dict, stages: dict, start: float) -> str:
        """Wait for the new champion to answer; record retrain-to-redeploy time (D37)."""
        candidates = (self.read_json("models/candidates.json") or {}).get("candidates", [])
        run_id = next(
            (c["run_id"] for c in candidates if c["variant"] == result.get("variant")), None
        )
        waited = self.served(run_id) if run_id else None
        if waited is None:
            stages["serving"] = {"status": "failed"}
            return "The model server hadn't switched to it within 3 minutes."
        total = round(time.monotonic() - start, 1)
        stages["serving"] = {"status": "ok", "seconds": waited}
        metrics.RETRAIN_TO_REDEPLOY.observe(total)
        return f"Serving it {waited:.0f} s after registration, {total:.0f} s after the run began."

    def repro(self, args: list[str], stages: dict, trigger: str) -> DvcResult:
        """Run a dvc command with the run's trigger, and merge its stage results into `stages`.

        A stage that ran earlier in this run keeps its result if a later command skips it.
        """
        result = self.dvc(args, self.repo, {**self.env, "SPATIFY_TRIGGER": trigger})
        for name, outcome in result.stages.items():
            if outcome["status"] != "skipped" or name not in stages:
                stages[name] = outcome
            if "seconds" in outcome:
                metrics.STAGE_SECONDS.labels(name).observe(outcome["seconds"])
        return result

    def after_failed_data(self, run_id: int, changes: Changes, result: DvcResult) -> str:
        """Quarantine the batches validation rejected; mark others failed, so no retry loop."""
        stage = failed_stage(result)
        if stage == "validate":
            report = self.read_json("metrics/validate.json") or {}
            rates = report.get("reject_rate_by_source", {})
            bad = [
                s for s in report.get("failed_sources", []) if (self.repo / INCOMING / s).exists()
            ]
            if bad:
                (self.repo / QUARANTINE).mkdir(exist_ok=True)
                for name in bad:
                    shutil.move(self.repo / INCOMING / name, self.repo / QUARANTINE / name)
                    self.store.set_batch(
                        name, "quarantined", run_id,
                        message=f"{rates.get(name, 0):.1%} of rows were invalid",
                    )  # fmt: skip
                metrics.QUARANTINED.inc(len(bad))
                self.dvc(["add", str(INCOMING)], self.repo, self.env)  # the folder without them
                listed = ", ".join(f"{s} ({rates.get(s, 0):.1%} of rows invalid)" for s in bad)
                return (
                    f"Validation rejected {listed}; moved to data/quarantine/. The champion keeps "
                    "serving. Any other new batches are checked again on the next run."
                )
        for name in changes.new:
            self.store.set_batch(name, "failed", run_id, message=f"the {stage} stage failed")
        return (
            f"The {stage} stage failed: {error_line(result.tail)}. The champion keeps serving. "
            + RETRY
        )

    def record_drift(self, run_id: int, changes: Changes) -> list[str]:
        """Mark the new batches processed with their drift result; the drifted ones' names."""
        report = (self.read_json("metrics/drift.json") or {}).get("batches", {})
        drifted, latest = [], None
        for name in changes.new:
            batch = report.get(name)
            is_drifted = bool(batch and batch["drifted"])
            self.store.set_batch(name, "processed", run_id, drifted=is_drifted if batch else None)
            if is_drifted:
                drifted.append(name)
            if batch:
                latest = batch
        if latest is not None:  # the gauges show the most recent batch
            for feature, values in latest["features"].items():
                metrics.DRIFT_PSI.labels(feature).set(values["psi"])
            metrics.DRIFT_DETECTED.set(1 if latest["drifted"] else 0)
        return drifted

    def read_json(self, relative: str) -> dict | None:
        try:
            return json.loads((self.repo / relative).read_text())
        except (OSError, ValueError):
            return None


RETRY = "Fix the cause, then restart the worker (docker compose restart worker) to retry."


def error_line(tail: list[str]) -> str:
    """DVC's ERROR line from its output (its last line is a generic sign-off)."""
    errors = [line for line in tail if line.startswith("ERROR")]
    if errors:
        return errors[-1].removeprefix("ERROR: ").strip()
    lines = [line for line in tail if line.strip() and "dvc.org/support" not in line]
    return lines[-1] if lines else "no output"


def failed_stage(result: DvcResult) -> str:
    return next((n for n, s in result.stages.items() if s["status"] == "failed"), "pipeline")
