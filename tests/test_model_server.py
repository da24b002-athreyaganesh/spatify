"""The model-server supervisor follows the champion alias (no real MLflow or processes)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from spatify.model_server import RegistryUnavailableError, Supervisor, alias_version

# What MLflow's REST API answers for an alias lookup (checked against the running server).
ANSWERS = {
    ("spatify", "champion"): (200, {"model_version": {"version": "3"}}),
    ("spatify", "nope"): (
        400,
        {
            "error_code": "INVALID_PARAMETER_VALUE",
            "message": "Registered model alias nope not found.",
        },
    ),
    ("missing", "champion"): (
        404,
        {
            "error_code": "RESOURCE_DOES_NOT_EXIST",
            "message": "Registered Model with name=missing not found",
        },
    ),
    ("broken", "champion"): (500, {"error_code": "INTERNAL_ERROR", "message": "database is down"}),
}


@pytest.fixture(scope="module")
def mlflow_api():
    """A tiny HTTP server answering /api/2.0/mlflow/registered-models/alias like MLflow."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = parse_qs(urlparse(self.path).query)
            status, body = ANSWERS[(query["name"][0], query["alias"][0])]
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_alias_lookup_reads_the_version(mlflow_api):
    assert alias_version(mlflow_api, "spatify", "champion") == "3"


def test_no_alias_or_no_model_means_nothing_to_serve_yet(mlflow_api):
    assert alias_version(mlflow_api, "spatify", "nope") is None
    assert alias_version(mlflow_api, "missing", "champion") is None


def test_registry_errors_are_reported_not_mistaken_for_no_champion(mlflow_api):
    with pytest.raises(RegistryUnavailableError):
        alias_version(mlflow_api, "broken", "champion")
    with pytest.raises(RegistryUnavailableError):
        alias_version("http://127.0.0.1:9", "spatify", "champion", timeout=2)  # nothing listening


class FakeProcess:
    def __init__(self, version):
        self.version, self.code = version, None

    def poll(self):
        return self.code

    def terminate(self):
        self.code = 0

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


def make(alias):
    launched = []

    def launch(version):
        launched.append(FakeProcess(version))
        return launched[-1]

    def resolve():
        if isinstance(alias["v"], Exception):
            raise alias["v"]
        return alias["v"]

    return Supervisor(resolve, launch), launched


def test_waits_until_a_champion_exists():
    alias = {"v": None}
    supervisor, launched = make(alias)
    supervisor.step()
    assert launched == []
    alias["v"] = "1"
    supervisor.step()
    assert [p.version for p in launched] == ["1"]


def test_restarts_only_when_the_alias_moves():
    alias = {"v": "1"}
    supervisor, launched = make(alias)
    supervisor.step()
    supervisor.step()
    assert len(launched) == 1  # same version: keep serving
    alias["v"] = "2"  # promotion or rollback
    supervisor.step()
    assert [p.version for p in launched] == ["1", "2"]
    assert launched[0].code == 0  # the old server was stopped first


def test_restarts_a_crashed_server():
    alias = {"v": "1"}
    supervisor, launched = make(alias)
    supervisor.step()
    launched[0].code = 1  # crashed
    supervisor.step()
    assert [p.version for p in launched] == ["1", "1"]


def test_keeps_serving_while_the_registry_is_down():
    alias = {"v": "1"}
    supervisor, launched = make(alias)
    supervisor.step()
    alias["v"] = RegistryUnavailableError("connection refused")
    supervisor.step()
    assert len(launched) == 1 and launched[0].code is None
