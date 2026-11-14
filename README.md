# Spatify

Cold-start music recommendation with feedback-driven retraining and drift monitoring. DA3408 (AI Operations) course project.

The problem statement and course guidelines are in `docs/course/`, the design decisions in `docs/decision_log.md`, and the dataset notes in `docs/data.md`.

## Setup

Needs Docker with Compose, Git LFS and uv.

    git clone <repo-url> spatify && cd spatify
    uv sync --all-extras        # Python 3.12 environment from uv.lock
    uv run pre-commit install
    export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0   # Ray stages hang under `uv run` without it (D15)
    uv run dvc pull             # needs the DVC remote; without it, rebuild from the public source:
    uv run dvc update data/raw/spotify_tracks.csv.dvc && uv run dvc repro

## Run the serving stack

    uv run python scripts/make_env.py        # once: .env with random secrets
    docker compose up -d --build             # postgres, mlflow, model-server, gateway, frontend

    # First start only: train into the stack's MLflow, so there is a champion to serve.
    export MLFLOW_TRACKING_URI=http://localhost:5000
    uv run dvc repro --force --downstream train

    uv run python scripts/smoke_test.py      # end-to-end check through the gateway

- The web app (Discover, the operator console, Help): http://localhost:8501
- New data: drop a batch CSV into `data/incoming/`. Within about 15 s the worker validates it, checks it for drift and retrains if it has drifted; the console's Pipeline and Data drift tabs show the run. A batch that fails validation is moved to `data/quarantine/`.
- The worker updates `dvc.lock` and `data/incoming.dvc` but doesn't commit; commit them yourself. Don't run `dvc` on the host while the worker is running (DVC locks the repo).
- Gateway API, with interactive docs: http://localhost:8000/docs
- MLflow UI (runs, model versions, aliases): http://localhost:5000
- Admin calls (rollback, pipeline trigger) need the `X-API-Key` header; the key is `SPATIFY_API_KEY` in `.env`.
- `docker compose down` stops the stack; the database and the models stay in Docker volumes.
- Without `MLFLOW_TRACKING_URI` the pipeline logs to a local `mlflow.db` instead of the stack.
- Memory: the stack uses about 1.2 GB, and each service has a hard limit (D30). The Ray stages use 4 CPUs and a 1 GB object store; on a bigger machine raise `SPATIFY_RAY_CPUS` and `SPATIFY_RAY_OBJECT_STORE_MB`. `python scripts/peak_memory.py -- <command>` reports a command's peak memory and stops it if the machine runs low.

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
