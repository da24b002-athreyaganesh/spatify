"""Run the worker:  python -m spatify.worker   (the Compose service runs it in /repo)

Settings: DATABASE_URL, MLFLOW_TRACKING_URI, REPO_DIR (/repo), POLL_SECONDS (15),
METRICS_PORT (9102), SPATIFY_RAY_CPUS and SPATIFY_RAY_OBJECT_STORE_MB (the Ray caps, D30).
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from pathlib import Path

from prometheus_client import start_http_server

from spatify.worker.store import WorkerStore
from spatify.worker.worker import Worker

log = logging.getLogger("spatify.worker")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    repo = Path(os.environ.get("REPO_DIR", "/repo"))
    poll = float(os.environ.get("POLL_SECONDS", "15"))
    store = WorkerStore(os.environ["DATABASE_URL"])
    for attempt in range(30):  # Postgres is healthy before we start, but be patient anyway
        try:
            store.init_schema()
            break
        except Exception:
            log.warning("Postgres not ready (attempt %d); retrying", attempt + 1)
            time.sleep(2)
    else:
        raise SystemExit("Postgres stayed unreachable")
    if interrupted := store.fail_interrupted_runs():
        log.warning("Marked %d run(s) left running by a previous worker as failed", interrupted)
    if retried := store.retry_failed_batches():
        log.info("Retrying %d batch(es) that failed before this restart", retried)
    start_http_server(int(os.environ.get("METRICS_PORT", "9102")))

    worker = Worker(repo, store)
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())  # finish the current run, then stop
    log.info("Watching %s and the run queue every %.0f s", repo / "data/incoming", poll)
    while not done.is_set():
        try:
            worked = worker.step()
        except Exception:  # e.g. Postgres briefly down: log, wait, try again
            log.exception("Check failed")
            worked = False
        if not worked or worker.last_failed:  # never retry a failure at full speed
            done.wait(poll)


if __name__ == "__main__":
    main()
