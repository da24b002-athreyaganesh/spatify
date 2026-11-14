"""Helpers shared by the pipeline stages: params, logging, Ray start-up and metrics."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import ray
import yaml
from ray.data import DataContext


def load_params(section: str, path: Path = Path("params.yaml")) -> dict:
    """Return one section of params.yaml."""
    return yaml.safe_load(path.read_text())[section]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def init_ray() -> None:
    """Start a local Ray instance, or join the cluster in RAY_ADDRESS (e.g. "auto").

    Joining a running cluster (`ray start --head`) keeps the Ray Dashboard up after
    the stage ends, which is what you want for screenshots.
    """
    if not ray.is_initialized():
        address = os.environ.get("RAY_ADDRESS")
        if address:
            ray.init(address=address, log_to_driver=False, logging_level="ERROR")
        else:
            ray.init(include_dashboard=False, log_to_driver=False, logging_level="ERROR")
    logging.getLogger("ray.data").setLevel(logging.WARNING)  # hide per-dataset INFO lines
    ctx = DataContext.get_current()
    ctx.enable_progress_bars = False
    # Same input, same row order: keeps outputs reproducible from run to run.
    ctx.execution_options.preserve_order = True


def fresh_dir(path: Path) -> Path:
    """Delete and recreate an output folder, so no files from an earlier run survive."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def write_metrics(path: Path, metrics: dict) -> None:
    """Write a DVC metrics file (JSON), creating its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")


def throughput(rows: int, seconds: float) -> dict:
    """Rows, wall-clock seconds and rows per second, rounded for the metrics files."""
    return {
        "rows": rows,
        "seconds": round(seconds, 3),
        "rows_per_s": round(rows / seconds, 1) if seconds > 0 else None,
    }
