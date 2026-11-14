"""The pipeline as the console shows it: DVC stages, the data flowing between them, each
stage's last result and its throughput (D34).

The gateway reads `dvc.yaml`, `dvc.lock` and `metrics/` from a read-only mount of the repo
(PIPELINE_DIR). A stage's result comes from the latest pipeline run the worker recorded,
if it reported one for that stage; otherwise a stage recorded in dvc.lock counts as last
run successfully, and one missing from it as never run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml

# Numbers from a stage's metrics file worth showing next to it.
SHOWN_METRICS = ("rows", "seconds", "rows_per_s", "tracks", "rejected", "reject_rate")


class PipelineUnavailableError(Exception):
    """The pipeline definition isn't mounted into the gateway (it answers 503)."""


def _paths(entries: Iterable | None) -> Iterator[str]:
    """Paths from a dvc.yaml list, whose entries are strings or {path: options}."""
    for entry in entries or []:
        yield entry if isinstance(entry, str) else next(iter(entry))


def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def load_dag(root: Path, latest_run: dict | None = None) -> dict:
    """Stages (in dvc.yaml order), external data inputs and the edges between them."""
    try:
        stages = yaml.safe_load((root / "dvc.yaml").read_text())["stages"]
    except (OSError, KeyError, TypeError) as err:
        raise PipelineUnavailableError(f"no readable dvc.yaml in {root}") from err
    try:
        locked = yaml.safe_load((root / "dvc.lock").read_text()).get("stages", {})
    except (OSError, AttributeError):
        locked = {}
    reported = (latest_run or {}).get("stages") or {}

    produced = {
        name: [*_paths(s.get("outs")), *_paths(s.get("metrics")), *_paths(s.get("plots"))]
        for name, s in stages.items()
    }

    def producer(path: str) -> str | None:
        return next(
            (n for n, outs in produced.items() if any(_inside(path, o) for o in outs)), None
        )

    edges, inputs = set(), set()
    for name, stage in stages.items():
        for dep in _paths(stage.get("deps")):
            source = producer(dep)
            if source is not None and source != name:
                edges.add((source, name))
            elif source is None and dep.startswith("data/"):  # data, not code or config
                inputs.add(dep)
                edges.add((dep, name))

    views = []
    for name, stage in stages.items():
        if isinstance(reported.get(name), dict) and reported[name].get("status"):
            status = reported[name]["status"]
        else:
            status = "ok" if name in locked else "not run"
        metrics = {}
        for path in _paths(stage.get("metrics")):
            try:
                values = json.loads((root / path).read_text())
            except (OSError, ValueError):
                continue
            metrics.update(
                {
                    k: float(values[k])
                    for k in SHOWN_METRICS
                    if isinstance(values.get(k), int | float)
                }
            )
        command = stage.get("cmd", "")
        views.append(
            {
                "name": name,
                "status": status,
                "command": " ".join(command) if isinstance(command, list) else str(command),
                "metrics": metrics,
            }
        )
    return {"stages": views, "inputs": sorted(inputs), "edges": sorted(edges)}
