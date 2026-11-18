"""Frontend tests: each page run headless with Streamlit's AppTest, against a fake gateway.

The fake answers the gateway's REST endpoints and records every call, so the tests check
both what the page shows and what it sends.
"""

import json
from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from spatify.frontend import api, ui

VIEWS = Path(__file__).parents[1] / "src/spatify/frontend/views"
KEY = "secret"

DAG = {
    "stages": [
        {
            "name": "ingest",
            "status": "ok",
            "command": "python -m x",
            "metrics": {"rows_per_s": 20000.0, "seconds": 4.0, "rows": 80000.0},
        },
        {"name": "train", "status": "failed", "command": "mlflow run", "metrics": {}},
    ],
    "inputs": ["data/incoming"],
    "edges": [["data/incoming", "ingest"], ["ingest", "train"]],
}
VERSIONS = [
    {
        "version": 2,
        "aliases": ["champion"],
        "run_id": "r2",
        "created_at": "2026-11-14T10:00:00Z",
        "description": "v2",
        "variant": "all-fisher",
        "dataset_version": "abcdef123",
        "git_commit": "0348807abc",
        "trigger": "manual",
        "likes": 3,
        "skips": 1,
        "like_rate": 0.75,
    },
    {
        "version": 1,
        "aliases": ["previous"],
        "run_id": "r1",
        "created_at": "2026-11-13T10:00:00Z",
        "description": "v1",
        "variant": "all-fisher",
        "dataset_version": "abcdef123",
        "git_commit": "0348807abc",
        "trigger": "manual",
        "likes": 0,
        "skips": 0,
        "like_rate": None,
    },
]


class FakeGateway:
    def __init__(self):
        self.calls, self.down, self.drift, self.explore = [], False, None, False
        self.alerts = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused")
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else None
        self.calls.append((method, path, body, request.headers.get("x-api-key")))
        admin = request.headers.get("x-api-key") == KEY
        if path == "/tracks/search":
            q = request.url.params["q"]
            tracks = (
                []
                if q == "zzz"
                else [
                    {
                        "track_id": "s1",
                        "track_name": "Unholy",
                        "artists": "Sam Smith",
                        "album_name": "Gloria",
                        "genres": ["pop"],
                        "popularity": 99,
                    }
                ]
            )
            return httpx.Response(200, json={"query": q, "model_version": 2, "tracks": tracks})
        if path == "/recommend":
            recs = [
                {
                    "track_id": f"t{i}",
                    "track_name": f"Song {i}",
                    "artists": "Band",
                    "album_name": "LP",
                    "genres": ["pop"],
                    "popularity": 50,
                    "rank": i + 1,
                    "score": 0.9,
                    "why": ["energetic"],
                }
                for i in range(body["k"])
            ]
            if self.explore:  # the last slot is a random pick, as the gateway does (D37)
                recs[-1] |= {"why": [], "explored": True}
            return httpx.Response(
                200,
                json={
                    "request_id": "11111111-1111-1111-1111-111111111111",
                    "model_version": 2,
                    "model_run_id": "r2",
                    "recommendations": recs,
                },
            )
        if path == "/feedback":
            return httpx.Response(201, json={"status": "recorded", "model_version": 2})
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "checks": {
                        "postgres": "ok",
                        "model_server": "ok",
                        "registry": "champion is version 2",
                    },
                },
            )
        if path == "/pipeline/dag":
            return httpx.Response(200, json=DAG)
        if path == "/pipeline/status":
            return httpx.Response(200, json={"latest": None, "counts": {}})
        if path == "/pipeline/runs":
            return httpx.Response(200, json=[])
        if path == "/models":
            return httpx.Response(
                200,
                json={
                    "name": "spatify",
                    "champion": 2,
                    "previous": 1,
                    "challenger": None,
                    "versions": VERSIONS,
                },
            )
        if path == "/models/rollback":
            if not admin:
                return httpx.Response(401, json={"detail": "Missing or wrong X-API-Key header"})
            return httpx.Response(
                200, json={"champion": 1, "previous": 2, "detail": "Version 1 is now the champion."}
            )
        if path == "/pipeline/trigger":
            if not admin:
                return httpx.Response(401, json={"detail": "Missing or wrong X-API-Key header"})
            return httpx.Response(202, json={"id": 7, "status": "requested"})
        if path == "/alerts":
            return httpx.Response(200, json=self.alerts)
        if path == "/drift":
            if self.drift is None:
                return httpx.Response(404, json={"detail": "No drift report yet"})
            return httpx.Response(200, json=self.drift)
        return httpx.Response(404, json={"detail": "not found"})


@pytest.fixture
def gateway():
    fake = FakeGateway()
    api.TRANSPORT = httpx.MockTransport(fake.handle)
    st.cache_data.clear()
    yield fake
    api.TRANSPORT = None


def page(name: str) -> AppTest:
    at = AppTest.from_file(str(VIEWS / f"{name}.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def texts(elements) -> str:
    return " ".join(str(e.value) for e in elements)


def test_discover_from_a_seed_song_to_feedback(gateway):
    at = page("discover")
    assert at.button(key="go").disabled  # nothing chosen yet
    at.text_input(key="search").input("unholy").run()
    at.selectbox(key="seed-unholy").select_index(0).run()
    assert not at.button(key="go").disabled
    at.slider(key="k").set_value(3).run()
    at.button(key="go").click().run()
    assert not at.exception
    sent = next(c for c in gateway.calls if c[1] == "/recommend")
    assert sent[2] == {"track_id": "s1", "sliders": {}, "k": 3}
    assert "Chosen by model version 2" in texts(at.caption)
    assert "**Why:** energetic" in texts(at.markdown)

    like = f"like-{at.session_state['result']['request_id']}-t0"
    at.button(key=like).click().run()
    feedback = next(c for c in gateway.calls if c[1] == "/feedback")
    assert feedback[2]["action"] == "like" and feedback[2]["track_id"] == "t0"
    assert at.button(key=like).disabled  # can't like twice; Skip still works


def test_sliders_alone_are_enough(gateway):
    at = page("discover")
    at.toggle(key="use-energy").set_value(True).run()
    at.button(key="go").click().run()
    sent = next(c for c in gateway.calls if c[1] == "/recommend")
    assert sent[2]["sliders"] == {"energy": 0.7} and "track_id" not in sent[2]


def test_a_random_pick_is_shown_as_a_wildcard(gateway):
    gateway.explore = True
    at = page("discover")
    at.toggle(key="use-energy").set_value(True).run()
    at.button(key="go").click().run()
    shown = texts(at.markdown)
    assert shown.count("**Why:** energetic") == 9 and shown.count("**A wildcard:**") == 1


def test_no_matches_says_what_to_do(gateway):
    at = page("discover")
    at.text_input(key="search").input("zzz").run()
    assert "No songs match" in texts(at.info)


def test_gateway_down_is_a_friendly_message_not_a_crash(gateway):
    gateway.down = True
    at = page("discover")
    at.text_input(key="search").input("unholy").run()
    assert not at.exception
    assert "can't reach its server" in texts(at.error)


def test_console_shows_pipeline_models_and_drift(gateway):
    at = page("console")
    assert len(at.get("graphviz_chart")) == 1
    assert "No drift report yet" in texts(at.info)  # before the first batch
    assert "No pipeline runs recorded yet" in texts(at.info)
    assert at.button(key="rollback").disabled  # no key, not confirmed
    assert at.button(key="run-pipeline").disabled


def test_console_rollback_sends_the_admin_key(gateway):
    at = page("console")
    at.sidebar.text_input(key="admin_key").input(KEY)
    at.checkbox(key="rollback_confirm").check().run()
    at.button(key="rollback").click().run()
    sent = next(c for c in gateway.calls if c[1] == "/models/rollback")
    assert sent[2] == {"version": 1} and sent[3] == KEY
    assert "Version 1 is now the champion" in texts(at.success)


def test_console_wrong_key_is_explained(gateway):
    at = page("console")
    at.sidebar.text_input(key="admin_key").input("wrong").run()
    at.button(key="run-pipeline").click().run()
    assert "admin key isn't right" in texts(at.warning)


def test_help_page_renders(gateway):
    at = page("help")
    assert "Get recommendations in three steps" in texts(at.header)


def test_dag_diagram_colours_stages_by_result():
    dot = ui.dag_dot(DAG)
    assert '"ingest" -> "train";' in dot and '"data/incoming" -> "ingest";' in dot
    assert ui.GREEN in dot and ui.VERMILLION in dot  # ok and failed
    assert "20,000 rows/s" in dot


def test_titles_with_markdown_characters_show_as_written():
    assert ui.md("*NSYNC [Live]") == r"\*NSYNC \[Live\]"


def test_like_rate_chart_needs_feedback():
    assert ui.like_rate_chart([VERSIONS[1]]) is None
    assert ui.like_rate_chart(VERSIONS) is not None


def test_the_app_opens_on_discover_and_navigates(gateway):
    at = AppTest.from_file(str(VIEWS.parent / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception and at.title[0].value == "Discover music"
    for view, title in (("console", "Operator console"), ("help", "Help")):
        at.switch_page(f"views/{view}.py").run()
        assert not at.exception and at.title[0].value == title


DRIFT = {
    "reference_rows": 79970,
    "thresholds": {"psi_moderate": 0.1, "psi_major": 0.2, "min_moderate": 2},
    "batches": [
        {
            "name": "bad_batch.csv",
            "status": "quarantined",
            "first_seen": "2026-11-15T10:02:00Z",
            "message": "21.6% of rows were invalid",
            "rows": None,
            "drifted": None,
            "features": [],
        },
        {
            "name": "shifted_01.csv",
            "status": "processed",
            "first_seen": "2026-11-15T10:01:00Z",
            "message": None,
            "rows": 1960,
            "drifted": True,
            "features": [
                {
                    "feature": "energy",
                    "psi": 0.21,
                    "ks_stat": 0.2,
                    "ks_pvalue": 1e-20,
                    "level": "major",
                },
                {
                    "feature": "tempo",
                    "psi": 0.14,
                    "ks_stat": 0.1,
                    "ks_pvalue": 1e-9,
                    "level": "moderate",
                },
            ],
        },
    ],
}


def test_console_shows_each_batch_and_the_latest_drift(gateway):
    gateway.drift = DRIFT
    at = page("console")
    table = at.dataframe[-2].value  # the batches table, above the chosen batch's details
    assert list(table["batch"]) == ["bad_batch.csv", "shifted_01.csv"]
    assert list(table["result"]) == ["rejected (quarantined)", "checked"]
    assert at.selectbox(key="drift_batch").value["name"] == "shifted_01.csv"  # the latest scored
    assert "has drifted" in texts(at.warning)
    assert ui.drift_chart(DRIFT["batches"][1], DRIFT["thresholds"]) is not None


FEEDBACK_DRIFT = {
    "recent_events": 200,
    "recent_like_rate": 0.06,
    "baseline_events": 1000,
    "baseline_like_rate": 0.41,
    "drop": 0.35,
    "threshold": 0.15,
    "enough_feedback": True,
    "drifted": True,
}


def test_console_warns_when_the_like_rate_falls(gateway):
    gateway.drift = {"reference_rows": None, "thresholds": {}, "batches": [],
                     "feedback": FEEDBACK_DRIFT}  # fmt: skip
    at = page("console")
    assert "No drift report yet" in texts(at.info)  # no batches, but feedback is shown
    rates = {m.label: m.value for m in at.metric}
    assert rates["Like rate, latest 200 ratings"] == "6%"
    assert rates["Like rate, the 1,000 before"] == "41%"
    assert "dropped by 35 points" in texts(at.warning)

    gateway.drift["feedback"] = FEEDBACK_DRIFT | {"drifted": False, "enough_feedback": False}
    at = page("console")
    assert "dropped by" not in texts(at.warning)
    assert "once enough likes and skips" in texts(at.caption)


ALERT = {
    "fingerprint": "abc",
    "alertname": "FeedbackDrift",
    "status": "firing",
    "severity": "warning",
    "summary": "Listeners are liking fewer songs",
    "description": "The like rate has dropped by 15 points or more.",
    "starts_at": "2026-11-15T10:00:00Z",
    "ends_at": None,
    "received_at": "2026-11-15T10:00:10Z",
}


def test_monitoring_tab_says_when_nothing_is_firing_and_embeds_grafana(gateway):
    at = page("console")
    assert "No alerts are firing" in texts(at.success)
    frames = at.get("iframe")
    assert len(frames) == len(ui.GRAFANA_PANELS)
    assert all("/d-solo/spatify-" in str(f.proto) for f in frames)


def test_monitoring_tab_shows_firing_alerts_first_then_the_history(gateway):
    resolved = ALERT | {
        "fingerprint": "old",
        "alertname": "DataDrift",
        "status": "resolved",
        "summary": "New tracks differ",
        "ends_at": "2026-11-15T09:00:00Z",
    }
    gateway.alerts = [ALERT, resolved]
    at = page("console")
    assert "Listeners are liking fewer songs" in texts(at.warning)
    assert "No alerts are firing" not in texts(at.success)
    history = next(d.value for d in at.dataframe if "alertname" in d.value.columns)
    assert list(history["status"]) == ["firing", "resolved"]


def test_monitoring_tab_still_shows_grafana_when_alerts_are_unavailable(gateway):
    original = gateway.handle

    def no_alerts(request):
        if request.url.path == "/alerts":
            return httpx.Response(503, json={"detail": "The feedback store is unavailable"})
        return original(request)

    api.TRANSPORT = httpx.MockTransport(no_alerts)
    at = page("console")
    assert "feedback store is unavailable" in texts(at.error)
    assert len(at.get("iframe")) == len(ui.GRAFANA_PANELS)


def test_metrics_are_served_from_start_up_not_from_the_first_page_view(monkeypatch):
    """`python -m spatify.frontend` starts /metrics before Streamlit runs any page."""
    import socket
    import urllib.request

    from spatify.frontend import __main__ as entry
    from spatify.frontend import metrics

    with socket.socket() as probe:  # a free port
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    order = []
    monkeypatch.setenv("METRICS_PORT", str(port))
    monkeypatch.setattr(entry.cli, "main", lambda: order.append("streamlit") or 0)
    real = entry.start_metrics_server
    monkeypatch.setattr(entry, "start_metrics_server", lambda p: order.append("metrics") or real(p))
    with pytest.raises(SystemExit):
        entry.main()
    assert order == ["metrics", "streamlit"]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        assert b"spatify_frontend_" in response.read()
    assert metrics.start_metrics_server(port)  # app.py's later call: no "address in use"
