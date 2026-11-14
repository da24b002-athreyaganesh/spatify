"""The model-server supervisor follows the champion alias (no real MLflow or processes)."""

from mlflow.exceptions import MlflowException

from spatify.model_server import Supervisor


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
    alias["v"] = MlflowException("connection refused")
    supervisor.step()
    assert len(launched) == 1 and launched[0].code is None
