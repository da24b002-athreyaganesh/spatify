"""Failure test: kill a pipeline stage mid-run; the champion must keep serving (TC-REL-01).

The problem statement's reliability requirement: "failed pipeline runs must not disrupt
the serving model". This queues a retrain through the API (as the console's button does),
waits for the train stage to start inside the worker, and kills it (SIGKILL, as an
out-of-memory kill would). Meanwhile a simulated listener asks for recommendations
five times a second. Four checks:
1. the run is recorded as failed;
2. its train stage is the one marked failed;
3. the champion is the same version afterwards;
4. every recommendation during the test was answered, by the champion.

Run with the stack up and the worker idle, from spatify/ (2-4 minutes):
    uv run python scripts/failure_test.py 2>&1 | tee evidence/phase7_failure_test.log

The admin key comes from --key, SPATIFY_API_KEY or .env, and is never printed.
Exits 0 if every check passed, 1 if not.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import httpx

# Run inside the worker container: SIGKILL every `python -m <stage module>` process (how
# `mlflow run` starts a stage). Only an exact "-m <module>" pair matches, never a command
# line that merely mentions the module.
KILLER = """
import os, signal, sys
target = sys.argv[1]
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == os.getpid():
        continue
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\\0")
    except OSError:
        continue
    if any(flag == "-m" and module == target for flag, module in zip(argv, argv[1:])):
        os.kill(int(pid), signal.SIGKILL)
        print(pid)
"""


def admin_key(given: str | None) -> str:
    if given:
        return given
    if os.environ.get("SPATIFY_API_KEY"):
        return os.environ["SPATIFY_API_KEY"]
    for line in Path(".env").read_text().splitlines():
        if line.startswith("SPATIFY_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("No admin key: pass --key, set SPATIFY_API_KEY, or run from spatify/ (.env)")


def stamp(start: float) -> str:
    return f"[{time.monotonic() - start:6.1f} s]"


class Listener(threading.Thread):
    """Asks for recommendations every `period` seconds until stopped; counts the answers."""

    def __init__(self, url: str, period: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.url, self.period = url, period
        self.stop, self.answers, self.failures = threading.Event(), Counter(), []

    def run(self) -> None:
        body = {"sliders": {"energy": 0.7, "valence": 0.6}, "k": 10}
        with httpx.Client(base_url=self.url, timeout=10) as api:
            while not self.stop.wait(self.period):
                try:
                    answer = api.post("/recommend", json=body)
                    if answer.status_code == 200:
                        self.answers[answer.json()["model_version"]] += 1
                    else:
                        self.failures.append(f"HTTP {answer.status_code}")
                except httpx.HTTPError as err:
                    self.failures.append(type(err).__name__)


def find_run(api: httpx.Client, run_id: int) -> dict | None:
    runs = api.get("/pipeline/runs", params={"limit": 50}).json()
    return next((r for r in runs if r["id"] == run_id), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kill a stage mid-run; check serving.")
    parser.add_argument("--url", default="http://localhost:8000", help="the gateway")
    parser.add_argument("--stage", default="train", choices=["train", "evaluate", "register"])
    parser.add_argument("--key", default=None, help="the admin key (default: from .env)")
    parser.add_argument("--timeout", type=float, default=600, help="seconds for the whole test")
    args = parser.parse_args()
    key, start = admin_key(args.key), time.monotonic()
    api = httpx.Client(base_url=args.url, timeout=15)

    try:
        if api.get("/health").status_code != 200:
            sys.exit("The gateway isn't healthy; check `docker compose ps`.")
    except httpx.HTTPError as err:
        sys.exit(f"Can't reach the gateway at {args.url} ({err}); is the stack up?")
    if any(r["status"] in ("requested", "running") for r in api.get("/pipeline/runs").json()):
        sys.exit("A pipeline run is queued or running; wait for it to finish, then retry.")
    before = api.get("/models").json()["champion"]
    print(f"Failure test: kill the {args.stage} stage mid-run; the champion must keep serving.")
    print(f"{stamp(start)} Champion before: version {before}")

    listener = Listener(args.url)
    listener.start()
    queued = api.post("/pipeline/trigger", headers={"X-API-Key": key})
    if queued.status_code != 202:
        listener.stop.set()
        sys.exit(f"Couldn't queue a run: HTTP {queued.status_code} {queued.text}")
    run_id = queued.json()["id"]
    print(f"{stamp(start)} Queued run {run_id} (a manual retrain); the worker checks every 15 s")

    killed, run = None, None
    target = f"spatify.model.{args.stage}"
    while time.monotonic() - start < args.timeout:
        run = find_run(api, run_id)
        if run and run["status"] in ("succeeded", "failed"):
            break
        if run and run["status"] == "running" and killed is None:
            result = subprocess.run(
                ["docker", "compose", "exec", "-T", "worker", "python", "-c", KILLER, target],
                capture_output=True, text=True, timeout=30,
            )  # fmt: skip
            if result.stdout.strip():
                killed = result.stdout.split()
                print(f"{stamp(start)} Killed the {args.stage} stage (pid {', '.join(killed)})")
        time.sleep(1)
    time.sleep(3)  # a few more recommendations after the run ended
    listener.stop.set()
    listener.join()
    after = api.get("/models").json()["champion"]

    if run is None or run["status"] not in ("succeeded", "failed"):
        sys.exit(f"Run {run_id} didn't finish within {args.timeout:.0f} s; see the worker's log.")
    stage = (run["stages"] or {}).get(args.stage, {})
    print(f"{stamp(start)} Run {run_id} finished: {run['status']}. {run['message'] or ''}")
    print(f"{stamp(start)} Champion after: version {after}")
    total = sum(listener.answers.values()) + len(listener.failures)
    print(
        f"{stamp(start)} Listener: {total} requests, {len(listener.failures)} failed, "
        f"answered by {dict(listener.answers)}"
    )

    checks = {
        "the run is recorded as failed": run["status"] == "failed",
        f"its {args.stage} stage is the one marked failed": stage.get("status") == "failed",
        "the champion is the same version": after == before,
        "every recommendation was answered, by the champion": not listener.failures
        and set(listener.answers) <= {before},
    }
    if killed is None:
        print(f"The {args.stage} stage finished before it could be killed; run the test again.")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(checks.values()) and killed is not None
    print(
        f"Result: {'PASS' if passed else 'FAIL'} ({sum(checks.values())} of {len(checks)} checks)"
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
