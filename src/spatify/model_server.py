"""Model server: serve the registry's champion, and switch when the alias moves.

`mlflow models serve -m models:/spatify@champion` resolves the alias once, at start-up
(L10), so a promotion or rollback wouldn't reach a running server. This supervisor
resolves the alias to a version number itself, serves that exact version, re-reads the
alias every POLL_SECONDS and restarts the server when it points somewhere else.

A switch means a few seconds without a server; the gateway answers 503 with a friendly
message meanwhile, and its /health reports it. The old server is stopped and waited for
before the new one starts, so the new one can't race the old one for the port (L10's
"restart succeeds against the OLD process" gotcha).

Considered: `mlflow models build-docker`, which bakes one version into an image, so
every promotion or rollback would need an image build and a redeploy.

The supervisor reads the alias with one call to MLflow's REST API instead of importing
the mlflow package, which would cost about 140 MB of memory for a single lookup (D30).

Run (as the Compose service does):  python -m spatify.model_server
Settings: MLFLOW_TRACKING_URI, MODEL_NAME (spatify), MODEL_ALIAS (champion),
          PORT (8080), WORKERS (1), POLL_SECONDS (10)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

log = logging.getLogger("spatify.model_server")


class RegistryUnavailableError(Exception):
    """MLflow couldn't be asked, or answered with an unexpected error."""


def alias_version(tracking_uri: str, name: str, alias: str, timeout: float = 10) -> str | None:
    """The version an alias points to, from MLflow's REST API; None if there isn't one yet.

    A missing model answers 404 RESOURCE_DOES_NOT_EXIST and a missing alias 400
    INVALID_PARAMETER_VALUE "... not found"; both mean nothing to serve yet.
    """
    query = urllib.parse.urlencode({"name": name, "alias": alias})
    url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/registered-models/alias?{query}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return str(json.load(response)["model_version"]["version"])
    except urllib.error.HTTPError as err:
        try:
            body = json.load(err)
        except ValueError:
            body = {}
        missing = body.get("error_code") in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE")
        if missing and "not found" in body.get("message", ""):
            return None
        raise RegistryUnavailableError(f"MLflow answered HTTP {err.code}: {body}") from err
    except (urllib.error.URLError, OSError, ValueError, KeyError) as err:
        raise RegistryUnavailableError(f"couldn't read the alias from MLflow: {err}") from err


def launch_server(name: str, version: str, port: int, workers: int) -> subprocess.Popen:
    """Start `mlflow models serve` for one exact registry version."""
    command = [
        "mlflow", "models", "serve",
        "--model-uri", f"models:/{name}/{version}",
        "--env-manager", "local",  # the image already has the model's packages
        "--host", "0.0.0.0",
        "--port", str(port),
        "--workers", str(workers),
    ]  # fmt: skip
    return subprocess.Popen(command)


class Supervisor:
    def __init__(
        self,
        resolve: Callable[[], str | None],
        launch: Callable[[str], subprocess.Popen],
    ) -> None:
        self.resolve = resolve  # returns the version the alias points to, or None
        self.launch = launch
        self.version: str | None = None
        self.process: subprocess.Popen | None = None

    def step(self) -> None:
        """One check: (re)start the server if the alias moved or the server died."""
        try:
            wanted = self.resolve()
        except RegistryUnavailableError as err:
            log.warning("Registry unreachable (%s); keeping version %s", err, self.version)
            return
        if wanted is None:
            log.info("No version has the alias yet; waiting for the pipeline to promote one")
            return
        died = self.process is not None and self.process.poll() is not None
        if wanted == self.version and not died:
            return
        if died:
            log.error("Model server for version %s exited; restarting", self.version)
        else:
            log.info("Switching from version %s to version %s", self.version, wanted)
        self.stop()
        self.process = self.launch(wanted)
        self.version = wanted

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    name = os.environ.get("MODEL_NAME", "spatify")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    port = int(os.environ.get("PORT", "8080"))
    workers = int(os.environ.get("WORKERS", "1"))
    poll = float(os.environ.get("POLL_SECONDS", "10"))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    supervisor = Supervisor(
        lambda: alias_version(tracking_uri, name, alias),
        lambda v: launch_server(name, v, port, workers),
    )
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    log.info("Serving %s@%s on port %d; checking the alias every %.0f s", name, alias, port, poll)
    try:
        while not done.is_set():
            supervisor.step()
            done.wait(poll)
    finally:
        supervisor.stop()


if __name__ == "__main__":
    main()
