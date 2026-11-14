"""What the worker does on each check (D36).

1. A run requested from the console or the API (`pipeline_runs`, status 'requested'):
   bring the data up to date, then retrain (`dvc repro`; if training was up to date,
   force it, since someone asked for a retrain).
2. Otherwise, new, changed or removed batch files in data/incoming:
   `dvc add data/incoming`, then `dvc repro drift_check` (ingest, validate, drift check).
   Retrain only if a new batch has drifted (trigger "drift"); otherwise the data is
   recorded and the model left alone, as the problem statement's FR5 asks.
3. Otherwise, nothing.

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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from spatify.worker import metrics
from spatify.worker.runner import DvcResult, run_dvc

log = logging.getLogger("spatify.worker")

INCOMING = Path("data/incoming")
QUARANTINE = Path("data/quarantine")


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
    ) -> None:
        self.repo, self.store, self.dvc = repo, store, dvc
        self.env = dict(os.environ if env is None else env)
        self.last_failed = False  # the main loop waits a full poll after a failed run

    def step(self) -> bool:
        """One check. True if it ran the pipeline."""
        metrics.QUEUE.set(self.store.queue_length())
        run = self.store.claim_requested()
        if run is not None:
            self.handle(run, self.incoming_changes(), requested=True)
            return True
        changes = self.incoming_changes()
        if changes:
            self.handle(self.store.start_run("new_data"), changes, requested=False)
            return True
        return False

    def incoming_changes(self) -> Changes:
        """Batch files that are new, changed or still waiting, and ones that were removed."""
        known = self.store.batches()
        files = {p.name: sha256(p) for p in sorted((self.repo / INCOMING).glob("*.csv"))}
        changes = Changes()
        for name, digest in files.items():
            row = known.get(name)
            if row is None or row["sha256"] != digest or row["status"] == "new":
                changes.new[name] = digest
        changes.removed = [
            name for name, row in known.items()
            if name not in files and row["status"] in ("new", "processed")
        ]  # fmt: skip
        return changes

    def handle(self, run: dict, changes: Changes, requested: bool) -> None:
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
            if not (requested or drifted):
                notes.append(
                    f"Checked {', '.join(changes.new)}: no drift, so the model wasn't retrained."
                    if changes.new
                    else "Brought the data stages up to date; no new batches."
                )
                return
            if drifted and not requested:
                trigger = "drift"
                self.store.update_run(run_id, trigger=trigger)
                notes.append(f"Drift in {', '.join(drifted)}, so the model was retrained.")
            full = self.repro(["repro"], stages, trigger)
            if full.ok and requested and stages.get("train", {}).get("status") != "ok":
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
                notes.append(
                    f"Registered version {version}: promoted to champion."
                    if promoted
                    else f"Registered version {version}: not promoted (gain {result.get('gain')}); "
                    f"version {result.get('champion_version')} keeps serving."
                )
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
            log.info("Run %s (%s) %s in %.0f s. %s", run_id, trigger, status, seconds, message)

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
