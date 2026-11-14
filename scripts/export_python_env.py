"""Write python_env.yaml, the MLproject's environment, from uv.lock.

The pins come from the same lock file as the development environment and the Docker
images, so all three match (decision D2). The DVC stage `python_env` reruns this whenever
uv.lock changes.

Run:  python scripts/export_python_env.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

HEADER = "# Generated from uv.lock by scripts/export_python_env.py; don't edit by hand.\n"


def main() -> None:
    exported = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",  # read uv.lock as it is; never re-resolve
            "--no-dev",
            "--extra",
            "pipeline",
            "--no-emit-project",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    dependencies = [
        line.strip() for line in exported.splitlines() if line.strip() and not line.startswith("#")
    ]
    env = {
        "python": Path(".python-version").read_text().strip(),
        "build_dependencies": ["pip"],
        "dependencies": dependencies,
    }
    Path("python_env.yaml").write_text(HEADER + yaml.safe_dump(env, sort_keys=False))
    print(f"Wrote python_env.yaml with {len(dependencies)} pinned packages")


if __name__ == "__main__":
    main()
