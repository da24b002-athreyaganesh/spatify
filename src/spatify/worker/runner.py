"""Running DVC for the worker, and following which stages ran, how long, and how they ended."""

from __future__ import annotations

import logging
import re
import subprocess
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("spatify.worker.dvc")

RUNNING = re.compile(r"^Running stage '([^']+)'")
SKIPPED = re.compile(r"^Stage '([^']+)' (?:didn't change|is cached)")
FAILED = re.compile(r"failed to reproduce '([^']+)'")


@dataclass
class DvcResult:
    ok: bool
    stages: dict[str, dict] = field(default_factory=dict)  # stage -> {status, seconds}
    tail: list[str] = field(default_factory=list)  # the last lines, for error messages
    seconds: float = 0.0


def follow(lines: Iterable[str], clock: Callable[[], float] = time.monotonic) -> dict[str, dict]:
    """Stage results from `dvc repro` output: ok, failed or skipped, with seconds run.

    DVC prints "Running stage 'x':" when a stage starts, so a stage ends when the next one
    starts or the output ends; "failed to reproduce 'x'" marks it failed. A stage DVC
    skipped (unchanged, or restored from its run cache) is "skipped".
    """
    stages: dict[str, dict] = {}
    current, started = None, 0.0

    def close(status: str) -> None:
        stages[current] = {"status": status, "seconds": round(clock() - started, 1)}

    for line in lines:
        if match := RUNNING.match(line):
            if current is not None:
                close("ok")
            current, started = match[1], clock()
            stages[current] = {"status": "running"}
        elif match := SKIPPED.match(line):
            stages.setdefault(match[1], {"status": "skipped"})
        elif (match := FAILED.search(line)) and match[1] == current:
            close("failed")
            current = None
    if current is not None and stages[current]["status"] == "running":
        close("ok")
    return stages


def run_dvc(args: list[str], cwd: Path, env: Mapping[str, str]) -> DvcResult:
    """Run `dvc <args>`, logging its output line by line as it comes."""
    start = time.monotonic()
    tail: deque[str] = deque(maxlen=30)
    process = subprocess.Popen(
        ["dvc", *args],
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def lines():
        for line in process.stdout:
            line = line.rstrip()
            tail.append(line)
            log.info("%s", line)
            yield line

    stages = follow(lines())
    ok = process.wait() == 0
    if not ok and not any(s["status"] == "failed" for s in stages.values()):
        # DVC failed before or between stages (a lock held elsewhere, a bad dvc.yaml).
        for stage in stages.values():
            if stage["status"] == "running":
                stage["status"] = "failed"
    return DvcResult(ok, stages, list(tail), round(time.monotonic() - start, 1))
