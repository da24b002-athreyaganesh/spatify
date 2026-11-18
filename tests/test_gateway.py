"""Gateway tests: the real app, registry logic and model client against fake services.

The model server is an httpx mock speaking the MLflow scoring protocol, the registry is a
stub MLflow client, and the feedback store is in memory. The same flows run against the
real Compose stack in scripts/smoke_test.py.
"""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from prometheus_client import REGISTRY

from spatify.gateway.app import create_app
from spatify.gateway.feedback_drift import feedback_drift
from spatify.gateway.model_client import ModelClient
from spatify.gateway.registry import Registry
from spatify.gateway.settings import Settings

KEY = "test-admin-key"


class FakeStore:
    def __init__(self):
        self.lists, self.feedback, self.runs = {}, {}, []
        self.up = True

    def open(self):
        pass

    def close(self):
        pass

    def ping(self):
        return self.up

    def save_recommendation(
        self, request_id, seed, sliders, k, version, run_id, track_ids, explored=None
    ):
        self.lists[request_id] = {
            "track_ids": track_ids,
            "model_version": version,
            "model_run_id": run_id,
            "explored": explored or [],
        }

    def like_rates(self, recent, baseline):
        """Like rates over the feedback recorded so far, newest first."""
        actions = [row["action"] for row in reversed(list(self.feedback.values()))]
        recent_part, base_part = actions[:recent], actions[recent : recent + baseline]

        def rate(part):
            return sum(a == "like" for a in part) / len(part) if part else None

        return {
            "recent_events": len(recent_part),
            "recent_rate": rate(recent_part),
            "baseline_events": len(base_part),
            "baseline_rate": rate(base_part),
        }

    def get_recommendation(self, request_id):
        return self.lists.get(request_id)

    def save_alerts(self, alerts):
        stored = getattr(self, "stored_alerts", {})
        for a in alerts:
            stored[(a["fingerprint"], a["starts_at"])] = {**a, "received_at": datetime.now(UTC)}
        self.stored_alerts = stored

    def alerts(self, limit):
        rows = sorted(
            getattr(self, "stored_alerts", {}).values(),
            key=lambda a: (a["status"] == "firing", a["received_at"]),
            reverse=True,
        )
        return rows[:limit]

    def save_feedback(self, request_id, track_id, rank, action, version, run_id):
        self.feedback[(request_id, track_id)] = {"rank": rank, "action": action, "version": version}

    def feedback_by_version(self):
        counts = {}
        for row in self.feedback.values():
            c = counts.setdefault(row["version"], {"likes": 0, "skips": 0})
            c["likes" if row["action"] == "like" else "skips"] += 1
        return counts

    def pipeline_runs(self, limit):
        return list(reversed(self.runs))[:limit]

    def pipeline_counts(self):
        return dict(Counter(r["status"] for r in self.runs))

    def incoming_batches(self):
        return getattr(self, "seen", [])

    def request_pipeline_run(self, trigger):
        row = {
            "id": len(self.runs) + 1,
            "requested_at": datetime.now(UTC),
            "started_at": None,
            "finished_at": None,
            "trigger": trigger,
            "status": "requested",
            "stages": {},
            "model_version": None,
            "promoted": None,
            "message": None,
        }
        self.runs.append(row)
        return row


class StubMlflowClient:
    """The registry calls the gateway makes, on an in-memory model with versions 1 and 2."""

    def __init__(self):
        self.aliases = {"champion": "2", "previous": "1"}
        self.tags = {}

    def get_registered_model(self, name):
        return SimpleNamespace(aliases=dict(self.aliases))

    def search_model_versions(self, filter_string):
        return [
            SimpleNamespace(
                version=str(v),
                run_id=f"run-{v}",
                creation_timestamp=1_763_000_000_000 + v,
                description=f"version {v}",
                tags={"variant": "all-fisher", "dataset_version": "abc", "trigger": "manual"},
            )
            for v in (1, 2)
        ]

    def get_model_version(self, name, version):
        if version not in ("1", "2"):
            raise MlflowException("not found")
        return SimpleNamespace(version=version)

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version

    def set_model_version_tag(self, name, version, key, value):
        self.tags[(version, key)] = value


class FakeModelServer:
    """Answers /ping and /invocations like `mlflow models serve`; `up = False` takes it down."""

    def __init__(self):
        self.up = True
        self.client = ModelClient("http://model-server", 1.0, httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("connection refused")
        if request.url.path == "/ping":
            return httpx.Response(200)
        body = json.loads(request.content)
        record, k = body["dataframe_records"][0], body["params"]["k"]
        if record.get("track_id") == "missing":
            message = "Failed to predict data. \nError: Unknown seed track_id: missing"
            return httpx.Response(
                400, json={"error_code": "INVALID_PARAMETER_VALUE", "message": message}
            )
        rows = [
            {
                "query": 0,
                "rank": i + 1,
                "track_id": f"t{i}",
                "track_name": f"Song {i}",
                "artists": "Band",
                "album_name": "Album",
                "genres": "pop;rock",
                "popularity": 50,
                "score": round(0.9 - i / 100, 4),
                "model_run_id": "run-2",
            }
            for i in range(k)
        ]
        return httpx.Response(200, json={"predictions": rows})


@pytest.fixture
def gw():
    store, server = FakeStore(), FakeModelServer()
    stub = StubMlflowClient()
    registry = Registry("http://mlflow:5000", "spatify", client=stub)
    catalogue = pd.DataFrame(
        {
            "track_id": ["a", "b", "c"],
            "track_name": ["Unholy", "Holy Water", "Other Song"],
            "artists": ["Sam Smith", "Band", "Unholy Choir"],
            "album_name": ["x", "y", "z"],
            "genres": ["pop", "rock", "choral"],
            "popularity": [100, 40, 10],
        }
    )
    registry.load_catalogue(2, catalogue)
    settings = Settings(
        api_key_sha256=hashlib.sha256(KEY.encode()).hexdigest(), refresh_seconds=3600
    )
    app = create_app(settings, store=store, registry=registry, model=server.client)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, store=store, server=server, mlflow=stub)


def test_recommend_by_seed_names_the_model_and_saves_the_list(gw):
    r = gw.client.post("/recommend", json={"track_id": "a", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == 2  # run-2 mapped through the registry
    assert [t["rank"] for t in body["recommendations"]] == [1, 2, 3]
    assert body["recommendations"][0]["genres"] == ["pop", "rock"]
    saved = gw.store.lists[next(iter(gw.store.lists))]
    assert saved["track_ids"] == ["t0", "t1", "t2"] and saved["model_version"] == 2
    assert "x-request-id" in r.headers


def test_recommend_by_sliders_only(gw):
    r = gw.client.post("/recommend", json={"sliders": {"energy": 0.9, "mode": 1}})
    assert r.status_code == 200 and len(r.json()["recommendations"]) == 10


@pytest.mark.parametrize(
    "body",
    [
        {},  # neither a seed nor a slider
        {"sliders": {}},
        {"sliders": {"energy": 1.5}},  # out of range
        {"sliders": {"tempo": -1}},
        {"sliders": {"loudnes": -5}},  # unknown slider
        {"track_id": "a", "k": 0},
        {"track_id": "a", "k": 51},
        {"track_id": ""},
    ],
)
def test_bad_requests_are_rejected_before_the_model(gw, body):
    assert gw.client.post("/recommend", json=body).status_code == 422


def test_unknown_seed_track_is_404(gw):
    r = gw.client.post("/recommend", json={"track_id": "missing"})
    assert r.status_code == 404
    assert "/tracks/search" in r.json()["detail"]


def test_model_server_down_gives_a_friendly_503_and_unhealthy(gw):
    gw.server.up = False
    r = gw.client.post("/recommend", json={"track_id": "a"})
    assert r.status_code == 503
    assert "Try again" in r.json()["detail"]
    health = gw.client.get("/health")
    assert health.status_code == 503
    assert health.json()["checks"]["model_server"] == "unreachable"


def test_health_needs_postgres_too(gw):
    assert gw.client.get("/health").status_code == 200
    gw.store.up = False
    assert gw.client.get("/health").status_code == 503


def test_feedback_is_tied_to_the_list_and_model(gw):
    shown = gw.client.post("/recommend", json={"track_id": "a", "k": 3}).json()
    rid = shown["request_id"]
    r = gw.client.post("/feedback", json={"request_id": rid, "track_id": "t1", "action": "like"})
    assert r.status_code == 201 and r.json()["model_version"] == 2
    assert next(iter(gw.store.feedback.values())) == {"rank": 2, "action": "like", "version": 2}

    not_shown = {"request_id": rid, "track_id": "zzz", "action": "skip"}
    assert gw.client.post("/feedback", json=not_shown).status_code == 400
    unknown = {
        "request_id": "00000000-0000-0000-0000-000000000000",
        "track_id": "t1",
        "action": "like",
    }
    assert gw.client.post("/feedback", json=unknown).status_code == 404
    bad_action = {"request_id": rid, "track_id": "t1", "action": "love"}
    assert gw.client.post("/feedback", json=bad_action).status_code == 422


def test_search_puts_title_matches_first(gw):
    r = gw.client.get("/tracks/search", params={"q": "unholy"})
    assert r.status_code == 200
    assert [t["track_id"] for t in r.json()["tracks"]] == ["a", "c"]  # title before artist
    assert gw.client.get("/tracks/search", params={"q": "u"}).status_code == 422


def test_models_show_aliases_and_like_rate(gw):
    shown = gw.client.post("/recommend", json={"track_id": "a", "k": 2}).json()
    for track, action in (("t0", "like"), ("t1", "skip")):
        gw.client.post(
            "/feedback",
            json={"request_id": shown["request_id"], "track_id": track, "action": action},
        )
    body = gw.client.get("/models").json()
    assert (body["champion"], body["previous"]) == (2, 1)
    v2 = next(v for v in body["versions"] if v["version"] == 2)
    assert v2["aliases"] == ["champion"] and v2["like_rate"] == 0.5


def test_rollback_needs_the_key_and_swaps_champion_and_previous(gw):
    assert gw.client.post("/models/rollback").status_code == 401
    wrong = {"X-API-Key": "nope"}
    assert gw.client.post("/models/rollback", headers=wrong).status_code == 401
    r = gw.client.post("/models/rollback", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    assert (r.json()["champion"], r.json()["previous"]) == (1, 2)
    assert gw.mlflow.aliases == {"champion": "1", "previous": "2"}
    again = gw.client.post("/models/rollback", headers={"X-API-Key": KEY}, json={"version": 1})
    assert again.status_code == 409  # already the champion


def test_pipeline_trigger_queues_a_run(gw):
    assert gw.client.post("/pipeline/trigger").status_code == 401
    r = gw.client.post("/pipeline/trigger", headers={"X-API-Key": KEY})
    assert r.status_code == 202 and r.json()["status"] == "requested"
    assert gw.client.get("/pipeline/runs").json()[0]["id"] == r.json()["id"]
    assert gw.client.get("/pipeline/status").json()["counts"] == {"requested": 1}


def test_metrics_count_requests_by_endpoint(gw):
    gw.client.post("/recommend", json={"track_id": "a"})
    text = gw.client.get("/metrics").text
    assert (
        'spatify_gateway_requests_total{endpoint="/recommend",method="POST",status="200"}' in text
    )
    assert 'spatify_recommendations_total{model_version="2"}' in text
    assert "spatify_model_server_request_seconds_bucket" in text
    assert "_created" not in text  # disable_created_metrics()


def test_drift_before_any_batch_is_404(gw):
    assert gw.client.get("/drift").status_code == 404


# --- Exploration and feedback drift (D37) ---------------------------------------------


def test_a_list_of_ten_gives_its_last_slot_to_a_random_track(gw):
    shown = gw.client.post("/recommend", json={"track_id": "a", "k": 10}).json()
    recs = shown["recommendations"]
    assert [r["rank"] for r in recs] == list(range(1, 11))
    assert [r["explored"] for r in recs] == [False] * 9 + [True]
    pick = recs[-1]
    assert pick["track_id"] in {"b", "c"}  # from the catalogue; not the seed, not a model pick
    assert pick["why"] == []
    saved = gw.store.lists[UUID(shown["request_id"])]
    assert saved["explored"] == [pick["track_id"]] and saved["track_ids"][-1] == pick["track_id"]
    rated = {"request_id": shown["request_id"], "track_id": pick["track_id"], "action": "skip"}
    assert gw.client.post("/feedback", json=rated).status_code == 201

    short = gw.client.post("/recommend", json={"track_id": "a", "k": 3}).json()
    assert not any(r["explored"] for r in short["recommendations"])  # int(0.1 * 3) = 0 slots


def test_drift_reports_the_like_rate_once_there_is_feedback(gw):
    shown = gw.client.post("/recommend", json={"track_id": "a", "k": 3}).json()
    for track_id, action in (("t0", "like"), ("t1", "skip")):
        rated = {"request_id": shown["request_id"], "track_id": track_id, "action": action}
        gw.client.post("/feedback", json=rated)
    report = gw.client.get("/drift").json()  # no batches yet, but feedback is enough
    assert report["batches"] == []
    feedback = report["feedback"]
    assert feedback["recent_events"] == 2 and feedback["recent_like_rate"] == 0.5
    assert not feedback["enough_feedback"] and not feedback["drifted"]


def test_feedback_drift_needs_a_big_enough_drop_and_enough_feedback():
    stats = {
        "recent_events": 200,
        "recent_rate": 0.1,
        "baseline_events": 1000,
        "baseline_rate": 0.4,
    }
    report = feedback_drift(stats, recent=200, baseline=1000, threshold=0.15)
    assert report["drop"] == 0.3 and report["enough_feedback"] and report["drifted"]
    small_drop = feedback_drift(stats | {"recent_rate": 0.3}, 200, 1000, 0.15)
    assert small_drop["drop"] == 0.1 and not small_drop["drifted"]
    too_few = feedback_drift(stats | {"recent_events": 50}, 200, 1000, 0.15)
    assert not too_few["enough_feedback"] and not too_few["drifted"]
    half_a_baseline = feedback_drift(stats | {"baseline_events": 500}, 200, 1000, 0.15)
    assert half_a_baseline["drifted"]  # the baseline needs only half its window
    nothing = feedback_drift({}, 200, 1000, 0.15)
    assert nothing["drop"] is None and nothing["recent_events"] == 0 and not nothing["drifted"]


# --- "Why" labels (D33) ---------------------------------------------------------------


def explainable_registry(run_id: str) -> Registry:
    """A registry whose champion catalogue holds the tracks the fake model server returns."""
    catalogue = pd.DataFrame(
        {
            "track_id": [f"t{i}" for i in range(6)],
            "track_name": [f"Song {i}" for i in range(6)],
            "artists": "Band",
            "album_name": "Album",
            "genres": "pop",
            "popularity": 50,
            "energy": [0.95, 0.9, 0.1, 0.5, 0.4, 0.6],
            "danceability": [0.8, 0.2, 0.5, 0.5, 0.6, 0.4],
        }
    )
    columns = ["energy", "danceability"]
    info = {
        "columns": columns,
        "mean": {c: float(catalogue[c].mean()) for c in columns},
        "std": {c: float(catalogue[c].std(ddof=0)) for c in columns},
        "weights": dict.fromkeys(columns, 1.0),
        "run_id": run_id,
    }
    registry = Registry("http://mlflow:5000", "spatify", client=StubMlflowClient())
    registry.load_catalogue(2, catalogue, info)
    return registry


def client_with(registry: Registry) -> TestClient:
    settings = Settings(
        api_key_sha256=hashlib.sha256(KEY.encode()).hexdigest(), refresh_seconds=3600
    )
    app = create_app(settings, store=FakeStore(), registry=registry, model=FakeModelServer().client)
    return TestClient(app)


def test_recommendations_say_why():
    with client_with(explainable_registry("run-2")) as client:  # run-2 answers, as loaded
        recs = client.post("/recommend", json={"sliders": {"energy": 0.9}, "k": 3}).json()
    why = {r["track_id"]: r["why"] for r in recs["recommendations"]}
    assert why == {"t0": ["energetic"], "t1": ["energetic"], "t2": []}  # t2 is calm


def test_no_labels_when_another_model_answered():
    # The gateway has version 1's settings loaded, but the champion just changed.
    with client_with(explainable_registry("run-1")) as client:
        recs = client.post("/recommend", json={"sliders": {"energy": 0.9}, "k": 2}).json()
    assert all(r["why"] == [] for r in recs["recommendations"])


# --- Pipeline view (D34) --------------------------------------------------------------

DVC_YAML = """
stages:
  make_batches:
    cmd: python -m spatify.batches
    deps: [data/raw/tracks.csv, src/spatify/batches.py]
    outs: [data/base]
  ingest:
    cmd: python -m spatify.pipeline.ingest
    deps: [data/base, data/incoming, src/spatify/pipeline/ingest.py]
    outs: [data/ingested]
    metrics:
    - metrics/ingest.json:
        cache: false
  features:
    cmd: python -m spatify.pipeline.features
    deps: [data/ingested]
    outs: [data/features]
  evaluate:
    cmd: mlflow run . -e evaluate
    deps: [data/features]
    metrics:
    - metrics/evaluate.json:
        cache: false
  register:
    cmd: mlflow run . -e register
    deps: [metrics/evaluate.json]
"""


@pytest.fixture
def pipeline_dir(tmp_path):
    (tmp_path / "dvc.yaml").write_text(DVC_YAML)
    (tmp_path / "dvc.lock").write_text(
        "schema: '2.0'\nstages:\n  make_batches: {}\n  ingest: {}\n  features: {}\n"
    )
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics/ingest.json").write_text(
        json.dumps({"rows": 79970, "seconds": 2.0, "rows_per_s": 39985.0, "files": 1})
    )
    return tmp_path


def dag_client(pipeline_dir, store=None) -> TestClient:
    settings = Settings(pipeline_dir=str(pipeline_dir), refresh_seconds=3600)
    registry = Registry("http://mlflow:5000", "spatify", client=StubMlflowClient())
    app = create_app(
        settings, store=store or FakeStore(), registry=registry, model=FakeModelServer().client
    )
    return TestClient(app)


def test_pipeline_dag_follows_the_data(pipeline_dir):
    with dag_client(pipeline_dir) as client:
        dag = client.get("/pipeline/dag").json()
    assert [s["name"] for s in dag["stages"]] == [
        "make_batches", "ingest", "features", "evaluate", "register"
    ]  # fmt: skip
    edges = {tuple(e) for e in dag["edges"]}
    assert ("make_batches", "ingest") in edges
    assert ("evaluate", "register") in edges  # through metrics/evaluate.json
    assert ("data/incoming", "ingest") in edges  # an input no stage produces
    assert dag["inputs"] == ["data/incoming", "data/raw/tracks.csv"]  # code deps left out
    status = {s["name"]: s["status"] for s in dag["stages"]}
    assert status == {
        "make_batches": "ok", "ingest": "ok", "features": "ok",
        "evaluate": "not run", "register": "not run",
    }  # fmt: skip
    ingest = next(s for s in dag["stages"] if s["name"] == "ingest")
    assert ingest["metrics"]["rows_per_s"] == 39985.0 and "files" not in ingest["metrics"]


def test_the_workers_report_overrides_dvc_lock(pipeline_dir):
    store = FakeStore()
    run = store.request_pipeline_run("manual")
    run["stages"] = {"features": {"status": "failed", "seconds": 3.1}}
    with dag_client(pipeline_dir, store) as client:
        dag = client.get("/pipeline/dag").json()
    assert {s["name"]: s["status"] for s in dag["stages"]}["features"] == "failed"


def test_pipeline_dag_without_the_mount_is_503(tmp_path):
    with dag_client(tmp_path / "missing") as client:
        assert client.get("/pipeline/dag").status_code == 503


# --- Drift (D35) ----------------------------------------------------------------------


def test_drift_lists_batches_newest_first_with_their_scores(pipeline_dir):
    report = {
        "reference_rows": 79970,
        "thresholds": {"psi_moderate": 0.1, "psi_major": 0.2, "min_moderate": 2},
        "batches": {
            "normal_01.csv": {"rows": 1960, "drifted": False, "features": {
                "energy": {"psi": 0.01, "ks_stat": 0.02, "ks_pvalue": 0.5, "level": "stable"}}},
            "shifted_01.csv": {"rows": 1960, "drifted": True, "features": {
                "energy": {"psi": 0.21, "ks_stat": 0.2, "ks_pvalue": 1e-20, "level": "major"}}},
        },
    }  # fmt: skip
    (pipeline_dir / "metrics/drift.json").write_text(json.dumps(report))
    store = FakeStore()
    store.seen = [  # as the worker records them, newest first
        {"name": "bad_batch.csv", "status": "quarantined", "first_seen": "2026-11-15T10:02:00Z",
         "drifted": None, "message": "21.6% of rows were invalid"},
        {"name": "shifted_01.csv", "status": "processed", "first_seen": "2026-11-15T10:01:00Z",
         "drifted": True, "message": None},
        {"name": "normal_01.csv", "status": "processed", "first_seen": "2026-11-15T10:00:00Z",
         "drifted": False, "message": None},
    ]  # fmt: skip
    with dag_client(pipeline_dir, store) as client:
        drift = client.get("/drift").json()
    assert [b["name"] for b in drift["batches"]] == [
        "bad_batch.csv",
        "shifted_01.csv",
        "normal_01.csv",
    ]
    bad, shifted, _ = drift["batches"]
    assert bad["features"] == [] and "invalid" in bad["message"]  # rejected: nothing to score
    assert shifted["drifted"] and shifted["features"][0]["level"] == "major"
    assert drift["reference_rows"] == 79970


# --- Alerts from Alertmanager, and the rank of rated tracks (D40, D41) -----------------


def alertmanager(status: str, ends: str = "0001-01-01T00:00:00Z") -> dict:
    """A webhook payload as Alertmanager sends it (abridged)."""
    return {
        "version": "4",
        "status": status,
        "receiver": "gateway",
        "alerts": [
            {
                "status": status,
                "labels": {"alertname": "DataDrift", "severity": "warning", "job": "worker"},
                "annotations": {"summary": "New tracks differ from the training data"},
                "startsAt": "2026-11-15T10:00:00Z",
                "endsAt": ends,
                "fingerprint": "abc123",
                "generatorURL": "http://prometheus:9090/graph",
            }
        ],
    }


def test_alerts_need_the_key_as_a_bearer_token(gw):
    body = alertmanager("firing")
    assert gw.client.post("/alerts", json=body).status_code == 401
    assert gw.client.post("/alerts", json=body, headers={"X-API-Key": KEY}).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert gw.client.post("/alerts", json=body, headers=wrong).status_code == 401
    assert gw.client.get("/alerts").json() == []


def test_an_alert_fires_then_resolves_as_one_record(gw):
    bearer = {"Authorization": f"Bearer {KEY}"}
    count = (
        REGISTRY.get_sample_value(
            "spatify_alert_notifications_total", {"alertname": "DataDrift", "status": "firing"}
        )
        or 0
    )
    assert gw.client.post("/alerts", json=alertmanager("firing"), headers=bearer).status_code == 204
    [firing] = gw.client.get("/alerts").json()
    assert firing["status"] == "firing" and firing["ends_at"] is None  # Alertmanager's year 1
    assert firing["alertname"] == "DataDrift" and firing["severity"] == "warning"
    assert (
        REGISTRY.get_sample_value(
            "spatify_alert_notifications_total", {"alertname": "DataDrift", "status": "firing"}
        )
        == count + 1
    )

    resolved = alertmanager("resolved", ends="2026-11-15T10:05:00Z")
    assert gw.client.post("/alerts", json=resolved, headers=bearer).status_code == 204
    [record] = gw.client.get("/alerts").json()  # the same alert, updated
    assert record["status"] == "resolved" and record["ends_at"].startswith("2026-11-15T10:05")


def test_firing_alerts_come_first(gw):
    bearer = {"Authorization": f"Bearer {KEY}"}
    old = alertmanager("resolved", ends="2026-11-15T10:05:00Z")
    new = alertmanager("firing")
    new["alerts"][0] |= {"fingerprint": "def456", "labels": {"alertname": "HighLatency"}}
    gw.client.post("/alerts", json=new, headers=bearer)
    gw.client.post("/alerts", json=old, headers=bearer)  # arrives later, but resolved
    assert [a["alertname"] for a in gw.client.get("/alerts").json()] == ["HighLatency", "DataDrift"]


def test_the_rank_of_each_rated_track_is_summarised(gw):
    def rank_sum():
        name = "spatify_feedback_rank_sum"
        return REGISTRY.get_sample_value(name, {"action": "like", "model_version": "2"}) or 0

    before = rank_sum()
    shown = gw.client.post("/recommend", json={"track_id": "a", "k": 3}).json()
    rated = {"request_id": shown["request_id"], "track_id": "t2", "action": "like"}
    assert gw.client.post("/feedback", json=rated).status_code == 201
    assert rank_sum() == before + 3  # t2 was third
