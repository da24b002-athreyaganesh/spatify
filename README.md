# Spatify

Cold-start music recommendation with feedback-driven retraining and drift monitoring. DA3408 (AI Operations) course project.

The problem statement and course guidelines are in `docs/course/`, the design decisions in `docs/decision_log.md`, and the dataset notes in `docs/data.md`.

## Setup

Needs Docker with Compose, Git LFS and uv.

    git clone <repo-url> spatify && cd spatify
    uv sync --all-extras        # Python 3.12 environment from uv.lock
    uv run pre-commit install
    uv run dvc pull             # needs the DVC remote; without it, rebuild from the public source:
    uv run dvc update data/raw/spotify_tracks.csv.dvc && uv run dvc repro

## Layout

| Path | Contents |
|---|---|
| `src/spatify/` | Python package: pipeline stages, model, shared code |
| `services/` | One folder per Compose service: gateway, model server, worker, frontend |
| `monitoring/` | Prometheus and Grafana configuration |
| `scripts/` | Operator scripts (user simulation, demo helpers) |
| `tests/` | pytest suite |
| `data/` | DVC-tracked data (`raw/`, `base/`, `batches/`); `incoming/` is the drop folder for new batches |
| `docs/` | Design documents, decision log, data card, course documents |
| `evidence/` | Screenshots, terminal logs and timings for the report |
