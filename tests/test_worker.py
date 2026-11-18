"""Worker tests: its decisions on every path, with a fake DVC and an in-memory store.

The fake DVC writes the metrics files the real stages write (validate.json, drift.json,
register.json) and returns the stage results, so the worker reads them as in the stack.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from prometheus_client import REGISTRY

from spatify.worker.runner import DvcResult, follow
from spatify.worker.worker import Worker, error_line


class FakeStore:
    def __init__(self, feedback=0):
        self.runs, self.batch_rows = [], {}
        self.feedback = [
            {"request_id": f"r{i // 10}", "created_at": pd.Timestamp("2026-11-15", tz="UTC"),
             "seed_track_id": "s1", "sliders": "{}", "track_id": f"t{i % 10}", "rank": i % 10 + 1,
             "action": "like" if i % 3 else "skip", "explored": i % 10 == 9, "model_version": 1}
            for i in range(feedback)
        ]  # fmt: skip

    def feedback_count(self):
        return len(self.feedback)

    def export_feedback(self):
        return self.feedback

    def queue_length(self):
        return sum(r["status"] == "requested" for r in self.runs)

    def request(self, trigger="manual"):
        self.runs.append({"id": len(self.runs) + 1, "trigger": trigger, "status": "requested"})

    def claim_requested(self):
        run = next((r for r in self.runs if r["status"] == "requested"), None)
        if run:
            run["status"] = "running"
        return run

    def start_run(self, trigger):
        self.runs.append({"id": len(self.runs) + 1, "trigger": trigger, "status": "running"})
        return self.runs[-1]

    def update_run(self, run_id, finished=False, **fields):
        self.runs[run_id - 1].update(fields, finished=finished)

    def batches(self):
        return {n: dict(r) for n, r in self.batch_rows.items()}

    def record_batch(self, name, sha256, run_id):
        self.batch_rows[name] = {"name": name, "sha256": sha256, "status": "new"}

    def set_batch(self, name, status, run_id, drifted=None, message=None):
        self.batch_rows[name].update(status=status, drifted=drifted, message=message)


class FakeDvc:
    """Plays DVC: `drifted` and `rejected` say how each batch fares."""

    def __init__(
        self,
        repo: Path,
        drifted=(),
        rejected=(),
        fail_stage=None,
        up_to_date=False,
        add_fails=False,
    ):
        self.repo, self.calls = repo, []
        self.drifted, self.rejected = set(drifted), set(rejected)
        self.fail_stage, self.up_to_date, self.add_fails = fail_stage, up_to_date, add_fails

    def write(self, name, data):
        (self.repo / "metrics").mkdir(exist_ok=True)
        (self.repo / "metrics" / name).write_text(json.dumps(data))

    def __call__(self, args, cwd, env):
        self.calls.append((args, env.get("SPATIFY_TRIGGER")))
        incoming = sorted(p.name for p in (self.repo / "data/incoming").glob("*.csv"))
        if args[0] == "add":
            if self.add_fails:  # what DVC printed when the container user had no account
                tail = [
                    "ERROR: unexpected error - 'getpwuid(): uid not found: 1000'",
                    "",
                    "Having any troubles? Hit us up at https://dvc.org/support, "
                    "we are always happy to help!",
                ]
                return DvcResult(False, {}, tail)
            return DvcResult(ok=True)
        ran = {"status": "ok", "seconds": 1.0}
        skipped = {"status": "skipped"}
        if args == ["repro", "drift_check"]:
            if self.fail_stage == "ingest":
                return DvcResult(False, {"ingest": {"status": "failed", "seconds": 1.0}}, ["boom"])
            bad = [n for n in incoming if n in self.rejected]
            self.write("validate.json", {
                "failed_sources": bad,
                "reject_rate_by_source": {n: 0.2155 if n in bad else 0.02 for n in incoming},
            })  # fmt: skip
            if bad:
                stages = {"ingest": ran, "validate": {"status": "failed", "seconds": 1.0}}
                return DvcResult(False, stages, ["ERROR: failed to reproduce 'validate'"])
            batches = {}
            for n in incoming:
                psi = 0.3 if n in self.drifted else 0.01
                batches[n] = {"drifted": n in self.drifted, "features": {"energy": {"psi": psi}}}
            self.write("drift.json", {"batches": batches})
            step = skipped if self.up_to_date else ran
            return DvcResult(True, {"ingest": step, "validate": step, "drift_check": step})
        if args == ["repro"]:
            if self.up_to_date:
                return DvcResult(True, {s: skipped for s in ("train", "evaluate", "register")})
            (self.repo / "models").mkdir(exist_ok=True)
            (self.repo / "models/candidates.json").write_text(
                json.dumps({"candidates": [{"variant": "all-feedback", "run_id": "run-5"}]})
            )
            self.write("register.json", {
                "version": 5, "promoted": True, "gain": 0.01, "champion_version": 5,
                "variant": "all-feedback", "metric": "feedback_ndcg",
            })  # fmt: skip
            return DvcResult(True, {s: ran for s in ("features", "train", "evaluate", "register")})
        if args == ["repro", "--force", "--downstream", "train"]:
            self.write(
                "register.json",
                {"version": 6, "promoted": False, "gain": 0.0, "champion_version": 5},
            )
            return DvcResult(True, {s: ran for s in ("train", "evaluate", "register")})
        raise AssertionError(f"unexpected dvc call {args}")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "data/incoming").mkdir(parents=True)
    return tmp_path


def drop(repo: Path, *names: str) -> None:
    for name in names:
        (repo / "data/incoming" / name).write_text(f"track_id\n{name}\n")


def make(repo, feedback=0, **dvc_options):
    store, dvc = FakeStore(feedback), FakeDvc(repo, **dvc_options)
    served = []
    worker = Worker(repo, store, dvc=dvc, env={}, served=lambda run: served.append(run) or 12.0)
    worker.served_calls = served
    return worker, store, dvc


def test_follow_reads_stage_results_from_dvc_output():
    clock = iter([0.0, 2.0, 2.0, 5.5]).__next__
    lines = [
        "Stage 'make_batches' didn't change, skipping",
        "Running stage 'ingest':",
        "> python -m spatify.pipeline.ingest",
        "Updating lock file 'dvc.lock'",
        "Running stage 'validate':",
        "ERROR: failed to reproduce 'validate': failed to run: python -m spatify.pipeline.validate",
    ]
    assert follow(lines, clock) == {
        "make_batches": {"status": "skipped"},
        "ingest": {"status": "ok", "seconds": 2.0},
        "validate": {"status": "failed", "seconds": 3.5},
    }


def test_nothing_new_means_nothing_runs(repo):
    worker, store, dvc = make(repo)
    assert worker.step() is False
    assert store.runs == [] and dvc.calls == []


def test_a_normal_batch_is_checked_but_the_model_is_not_retrained(repo):
    drop(repo, "normal_01.csv")
    worker, store, dvc = make(repo)
    assert worker.step() is True
    run = store.runs[0]
    assert (run["trigger"], run["status"]) == ("new_data", "succeeded")
    assert [c[0] for c in dvc.calls] == [["add", "data/incoming"], ["repro", "drift_check"]]
    assert "no drift" in run["message"]
    assert store.batch_rows["normal_01.csv"]["status"] == "processed"
    assert worker.step() is False  # already processed


def test_a_drifted_batch_retrains_with_trigger_drift(repo):
    drop(repo, "shifted_01.csv")
    worker, store, dvc = make(repo, drifted={"shifted_01.csv"})
    worker.step()
    run = store.runs[0]
    assert (run["trigger"], run["status"], run["model_version"], run["promoted"]) == (
        "drift", "succeeded", 5, True
    )  # fmt: skip
    assert (["repro"], "drift") in dvc.calls  # MLflow tags the new model trigger=drift
    assert run["stages"]["register"]["status"] == "ok"
    assert store.batch_rows["shifted_01.csv"]["drifted"] is True


def test_a_corrupt_batch_is_quarantined_and_the_champion_untouched(repo):
    drop(repo, "bad_batch.csv")
    worker, store, dvc = make(repo, rejected={"bad_batch.csv"})
    worker.step()
    run = store.runs[0]
    assert run["status"] == "failed" and "21.6% of rows invalid" in run["message"]
    assert run["stages"]["validate"]["status"] == "failed"
    assert (repo / "data/quarantine/bad_batch.csv").exists()
    assert not (repo / "data/incoming/bad_batch.csv").exists()
    assert store.batch_rows["bad_batch.csv"]["status"] == "quarantined"
    assert ["repro"] not in [c[0] for c in dvc.calls]  # register never ran
    assert worker.step() is False  # no retry loop


def test_good_batches_dropped_with_a_bad_one_go_through_next(repo):
    drop(repo, "bad_batch.csv", "normal_01.csv")
    worker, store, _ = make(repo, rejected={"bad_batch.csv"})
    worker.step()
    assert store.batch_rows["normal_01.csv"]["status"] == "new"  # waiting
    assert worker.step() is True  # a second run, without the bad batch
    assert store.runs[1]["status"] == "succeeded"
    assert store.batch_rows["normal_01.csv"]["status"] == "processed"


def test_a_failure_elsewhere_marks_the_batch_failed_so_it_is_not_retried(repo):
    drop(repo, "normal_01.csv")
    worker, store, _ = make(repo, fail_stage="ingest")
    worker.step()
    assert store.runs[0]["status"] == "failed" and "ingest stage failed" in store.runs[0]["message"]
    assert store.batch_rows["normal_01.csv"]["status"] == "failed"
    assert worker.step() is False


def test_a_requested_run_retrains_even_when_everything_is_up_to_date(repo):
    worker, store, dvc = make(repo, up_to_date=True)
    store.request("manual")
    worker.step()
    run = store.runs[0]
    assert run["status"] == "succeeded" and run["model_version"] == 6 and run["promoted"] is False
    assert (["repro", "--force", "--downstream", "train"], "manual") in dvc.calls
    assert "not promoted" in run["message"]


def test_a_removed_batch_brings_the_data_stages_up_to_date(repo):
    drop(repo, "normal_01.csv")
    worker, store, dvc = make(repo)
    worker.step()
    (repo / "data/incoming/normal_01.csv").unlink()
    assert worker.step() is True
    assert store.batch_rows["normal_01.csv"]["status"] == "removed"
    assert store.runs[1]["message"].startswith("Brought the data stages up to date")


def test_a_removed_batch_dropped_in_again_is_checked_again(repo):
    """The same file, same contents, back after removal: new data again (found in step 6.3)."""
    drop(repo, "shifted_01.csv")
    worker, store, dvc = make(repo, drifted={"shifted_01.csv"})
    worker.step()
    (repo / "data/incoming/shifted_01.csv").unlink()
    worker.step()
    assert store.batch_rows["shifted_01.csv"]["status"] == "removed"
    drop(repo, "shifted_01.csv")  # identical contents
    assert worker.step() is True
    assert store.runs[-1]["trigger"] == "drift"  # checked, found drifted, retrained
    assert store.batch_rows["shifted_01.csv"]["status"] == "processed"
    assert worker.step() is False  # and only once


def test_an_unexpected_error_fails_once_and_is_not_retried_at_full_speed(repo):
    drop(repo, "normal_01.csv")
    worker, store, _ = make(repo, add_fails=True)
    worker.step()
    run = store.runs[0]
    assert run["status"] == "failed"
    assert "getpwuid(): uid not found: 1000" in run["message"]  # the error, not DVC's sign-off
    assert "restart the worker" in run["message"]
    assert store.batch_rows["normal_01.csv"]["status"] == "failed"
    assert worker.last_failed  # the main loop waits a full poll before the next check
    assert worker.step() is False and len(store.runs) == 1  # no retry storm


def test_error_line_skips_dvcs_sign_off():
    assert (
        error_line(["ERROR: lock held", "", "Having any troubles? see https://dvc.org/support"])
        == "lock held"
    )
    assert (
        error_line(["some output", "Having any troubles? https://dvc.org/support"]) == "some output"
    )
    assert error_line([]) == "no output"


def test_enough_new_feedback_retrains_on_a_fresh_snapshot(repo):
    (repo / "params.yaml").write_text("feedback:\n  retrain_every: 50\n")
    worker, store, dvc = make(repo, feedback=60)
    assert worker.step() is True
    run = store.runs[0]
    assert (run["trigger"], run["status"], run["promoted"]) == ("feedback", "succeeded", True)
    snapshot = pd.read_parquet(repo / "data/feedback/feedback.parquet")
    assert len(snapshot) == 60 and snapshot["explored"].sum() == 6
    assert (["add", "data/feedback"], None) in dvc.calls  # the snapshot is versioned
    assert "Training saw 60 likes and skips" in run["message"]
    assert worker.step() is False  # nothing new since the snapshot


def test_too_little_new_feedback_waits(repo):
    (repo / "params.yaml").write_text("feedback:\n  retrain_every: 100\n")
    worker, store, _ = make(repo, feedback=60)
    assert worker.step() is False and store.runs == []


def test_a_promotion_waits_for_the_model_server_and_records_the_time(repo):
    drop(repo, "shifted_01.csv")
    worker, store, _ = make(repo, drifted={"shifted_01.csv"})
    worker.step()
    run = store.runs[0]
    assert worker.served_calls == ["run-5"]  # the promoted candidate's training run
    assert run["stages"]["serving"] == {"status": "ok", "seconds": 12.0}
    assert "Serving it 12 s after registration" in run["message"]


def test_stage_throughput_comes_from_the_stage_metrics_files(repo):
    def throughput(stage):
        return REGISTRY.get_sample_value("spatify_pipeline_stage_rows_per_second", {"stage": stage})

    (repo / "metrics").mkdir()
    (repo / "metrics/ingest.json").write_text(json.dumps({"rows": 100, "rows_per_s": 2500.0}))
    worker, _, _ = make(repo)
    assert throughput("ingest") == 2500.0  # known from the start, before any run

    (repo / "metrics/features.json").write_text(json.dumps({"rows": 100, "rows_per_s": 900.0}))
    drop(repo, "normal_01.csv")
    worker.step()  # a run; afterwards the gauges are read again
    assert throughput("features") == 900.0
    assert REGISTRY.get_sample_value("spatify_pipeline_stage_rows", {"stage": "ingest"}) == 100
