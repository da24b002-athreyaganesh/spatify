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

`mlflow models serve` has no /metrics, so the supervisor exports the server's side on
METRICS_PORT (D40): the version being served, every (re)start and why, how long a new
version takes to answer /ping (the switch's downtime), and the serving processes' memory.
The gateway measures the requests themselves.

Run (as the Compose service does):  python -m spatify.model_server
Settings: MLFLOW_TRACKING_URI, MODEL_NAME (spatify), MODEL_ALIAS (champion),
          PORT (8080), WORKERS (1), POLL_SECONDS (10), METRICS_PORT (9103)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, disable_created_metrics, start_http_server

log = logging.getLogger("spatify.model_server")

disable_created_metrics()
SERVING_VERSION = Gauge(
    "spatify_model_server_serving_version", "Registry version being served (0: none yet)"
)
LAUNCHES = Counter(
    "spatify_model_server_launches_total",
    "Serving process starts, by reason (start, alias_moved, crashed)",
    ["reason"],
)
READY_SECONDS = Histogram(
    "spatify_model_server_ready_seconds",
    "From starting a version to its first successful /ping: a switch's downtime",
    buckets=(2, 4, 6, 8, 10, 12, 15, 20, 30, 60, 120),
)
REGISTRY_ERRORS = Counter(
    "spatify_model_server_registry_errors_total", "Alias lookups that MLflow didn't answer"
)
PROCESS_UP = Gauge("spatify_model_server_process_up", "1 while the serving process runs")
RESIDENT_BYTES = Gauge(
    "spatify_model_server_resident_memory_bytes",
    "Resident memory of the serving process and its workers",
)


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


def ping(port: int, timeout: float = 2) -> bool:
    """True if the model server on `port` answers /ping."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def tree_rss(pid: int, proc: Path = Path("/proc")) -> int:
    """Resident memory in bytes of a process and all its descendants, read from /proc.

    `mlflow models serve` starts the web server as a child process, so the memory of the
    process the supervisor launched alone would miss the model.
    """
    parents = {}
    for stat in proc.glob("[0-9]*/stat"):
        try:  # "pid (name) state ppid ...": the name may hold spaces and brackets
            parents[int(stat.parent.name)] = int(stat.read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue  # the process ended while we looked
    family, frontier = {pid}, [pid]
    while frontier:
        children = [child for child, parent in parents.items() if parent in frontier]
        frontier = [child for child in children if child not in family]
        family.update(frontier)
    page = os.sysconf("SC_PAGE_SIZE")
    total = 0
    for member in family:
        try:
            total += int((proc / str(member) / "statm").read_text().split()[1]) * page
        except (OSError, IndexError, ValueError):
            continue
    return total


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
        ready: Callable[[], bool] | None = None,
        ready_timeout: float = 180,
    ) -> None:
        self.resolve = resolve  # returns the version the alias points to, or None
        self.launch = launch
        self.ready = ready  # True once the new server answers; times each (re)start
        self.ready_timeout = ready_timeout
        self.version: str | None = None
        self.process: subprocess.Popen | None = None
        PROCESS_UP.set_function(lambda: 1 if self.running() else 0)
        RESIDENT_BYTES.set_function(lambda: tree_rss(self.process.pid) if self.running() else 0)

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def step(self) -> None:
        """One check: (re)start the server if the alias moved or the server died."""
        try:
            wanted = self.resolve()
        except RegistryUnavailableError as err:
            REGISTRY_ERRORS.inc()
            log.warning("Registry unreachable (%s); keeping version %s", err, self.version)
            return
        if wanted is None:
            log.info("No version has the alias yet; waiting for the pipeline to promote one")
            return
        died = self.process is not None and self.process.poll() is not None
        if wanted == self.version and not died:
            return
        if died:
            reason = "crashed"
            log.error("Model server for version %s exited; restarting", self.version)
        else:
            reason = "start" if self.version is None else "alias_moved"
            log.info("Switching from version %s to version %s", self.version, wanted)
        self.stop()
        started = time.monotonic()
        self.process = self.launch(wanted)
        self.version = wanted
        LAUNCHES.labels(reason).inc()
        SERVING_VERSION.set(int(wanted))
        if self.ready is not None:
            threading.Thread(target=self.time_ready, args=(started,), daemon=True).start()

    def time_ready(self, started: float) -> float | None:
        """Wait for the new server to answer; record how long it took (None: it never did)."""
        while time.monotonic() - started < self.ready_timeout:
            if self.ready():
                seconds = time.monotonic() - started
                READY_SECONDS.observe(seconds)
                log.info("Version %s answering after %.1f s", self.version, seconds)
                return seconds
            time.sleep(0.5)
        log.error("Version %s didn't answer within %.0f s", self.version, self.ready_timeout)
        return None

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
    metrics_port = int(os.environ.get("METRICS_PORT", "9103"))

    supervisor = Supervisor(
        lambda: alias_version(tracking_uri, name, alias),
        lambda v: launch_server(name, v, port, workers),
        lambda: ping(port),
    )
    try:
        start_http_server(metrics_port)
    except OSError:  # serving matters more than its metrics
        log.warning("Couldn't serve metrics on port %d", metrics_port, exc_info=True)
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
