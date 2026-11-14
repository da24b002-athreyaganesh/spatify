"""What every MLflow run records beyond autolog: code, data and environment lineage."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

# Packages whose versions are logged with each run, and pinned in the model's requirements.
PACKAGES = ["mlflow", "numpy", "pandas", "pyarrow", "ray", "dvc"]
MODEL_PACKAGES = ["mlflow", "numpy", "pandas", "pyarrow"]


def _git(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_version(lock: Path = Path("dvc.lock"), output: str = "data/features") -> str:
    """The DVC hash of the feature data the model was trained on, from dvc.lock."""
    try:
        stages = yaml.safe_load(lock.read_text())["stages"]
    except (OSError, KeyError, TypeError):
        return "unknown"
    for stage in stages.values():
        for out in stage.get("outs", []):
            if out.get("path") == output:
                return out.get("md5", "unknown")
    return "unknown"


def lineage() -> dict[str, str]:
    """Tags linking a run to its code, data and reason for running.

    `git_dirty` says whether the working tree had uncommitted changes, in which case the
    commit alone doesn't describe the code. The trigger is set by whoever starts the
    pipeline (SPATIFY_TRIGGER: manual, new_data, drift or feedback; manual by default).
    """
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(_git("status", "--porcelain") not in ("", "unknown")).lower(),
        "dataset_version": data_version(),
        "trigger": os.environ.get("SPATIFY_TRIGGER", "manual"),
    }


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def environment() -> dict[str, str]:
    """Python, platform and key package versions."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{name: package_version(name) for name in PACKAGES},
    }


def model_requirements() -> list[str]:
    """Pinned pip requirements for the served model: only what the pyfunc imports.

    The versions are the installed ones, which come from uv.lock, so the model's
    environment matches development (L10: pin pip_requirements, the serving environment
    is rebuilt from them).
    """
    return [f"{name}=={version(name)}" for name in MODEL_PACKAGES]
