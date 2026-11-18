# Spatify



Cold-start music recommendation with feedback-driven retraining and drift monitoring: an end-to-end AIOps system. DA3408 (AI Operations) course project.

Spatify recommends songs that sound like a song you pick, or like the sliders you set, from audio features alone, so a new listener needs no history. Around that small model is the operational system the course is about:
- **Data:** a Ray Data pipeline validates new songs, and DVC versions every step.
- **Models:** MLflow tracks every model. A promotion gate lets only better models replace the champion, and a rollback undoes a bad one.
- **Automation:** a worker retrains by itself on new data, drift (PSI) or 2,000 new likes and skips.
- **Monitoring:** Prometheus, Grafana and Alertmanager watch every part.
- **The app:** a web page for listeners, plus a console for operators.

All of it runs as 11 Docker Compose services on one laptop.

**Demo video (3 minutes):** [watch it on Google Drive](https://drive.google.com/PASTE_THE_SHARE_LINK_HERE)

> **Most of the feedback is simulated.** Real listeners are few, so most likes and skips come from a documented user simulator (`scripts/simulate_users.py`, D38): simulated listeners with a hidden taste the system has to learn. The feedback-learning results show that the loop works, not what real listeners would gain.

## Documents

| Document | What it covers |
|---|---|
| [Architecture](docs/architecture.md) | Every Compose service, the data and model flow, the automation loops, and what each block does |
| [High-level design](docs/hld.md) | The design choices and the reasons for them, requirement traceability, limits |
| [Low-level design](docs/lld.md) | Every endpoint with its inputs and outputs (from the [OpenAPI spec](docs/openapi.json)), the model server's `/invocations`, the database tables, the pipeline, algorithms, metrics, configuration |
| [Test plan](docs/test_plan.md) and [test report](docs/test_report.md) | 15 acceptance criteria, 38 test cases, and the results |
| [User manual](docs/user_manual.md) | For a non-technical user: finding music, liking and skipping, the operator console |
| [Security note](docs/security.md) | What is protected and how |
| [Decision log](docs/decision_log.md) | Every design decision (D1–D44): what was chosen, why, what else was considered |
| [Data card](docs/data.md) | The dataset's source, licence and biases |
| [AI disclosure](AI_DISCLOSURE.md) | How AI tools were used, as the course's fair-use code requires |
| `docs/course/` | The problem statement and the course guidelines |

## Quick start

Needs Docker with Compose, Git LFS and [uv](https://docs.astral.sh/uv/), and about 3 GB of free memory.

```bash
git clone <repo-url> spatify && cd spatify
uv sync --all-extras                      # Python 3.12 environment from uv.lock
uv run pre-commit install
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0    # Ray stages hang under `uv run` without it (D15)
uv run dvc pull                           # the data; without the DVC remote, rebuild it:
#   uv run dvc update data/raw/spotify_tracks.csv.dvc && uv run dvc repro

uv run python scripts/make_env.py         # once: .env with random secrets
docker compose up -d --build              # all 11 services
docker compose ps                         # healthy or Up within about a minute

# First start only: train into the stack's MLflow, so there's a champion to serve.
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run dvc repro --force --downstream train

make smoke                                # end-to-end check through the gateway
```

| Open | Address |
|---|---|
| The web app (Discover, the operator console, Help) | http://localhost:8501 |
| The gateway's API, with interactive docs | http://localhost:8000/docs |
| MLflow (runs, model versions, aliases) | http://localhost:5000 |
| Grafana (Spatify folder; viewing needs no login) | http://localhost:3000 |
| Prometheus, Alertmanager | http://localhost:9090, http://localhost:9093 |

`docker compose stop` stops the stack and `docker compose start` brings it back. `docker compose down` removes the containers; the database, models and metrics stay in Docker volumes.

## The demo

The whole loop in 8 steps, with the stack up. Commands run from `spatify/`.

1. **Everything healthy.** Run `docker compose ps`, then open the console: the pipeline diagram and the last run's throughput.
2. **Cold-start recommendations.** On Discover, search for a song and get recommendations; point at the "why" labels and the model version.
3. **Feedback.** `uv run python scripts/simulate_users.py --lists 210`. The like rate shows in the console and Grafana. After 2,000 ratings the worker retrains, and the promotion gate decides.
4. **Drift.** `cp data/batches/shifted_01.csv data/incoming/`. PSI rises in the console and Grafana, the `DataDrift` alert fires, and the worker retrains.
5. **A new version.** If the gate promotes it, Discover's next list names the new version (the model server switches within about 15 s).
6. **A bad deployment.** `uv run python scripts/force_bad_model.py --yes`, then `uv run python scripts/simulate_users.py --lists 30`. The like rate collapses, and feedback drift fires.
7. **Rollback.** Console → Models → Roll back (admin key: `SPATIFY_API_KEY` in `.env`). The old version answers again within about 15 s.
8. **Lineage.** In MLflow, trace the served version to its run, git commit, DVC data version and trigger.

Afterwards: `rm data/incoming/shifted_01.csv`. The worker updates `dvc.lock` and `data/incoming.dvc` but doesn't commit them; commit them yourself.

## Working with the pipeline

- **Reproduce everything:** `uv run dvc repro` runs only the stages whose inputs changed; `uv run dvc dag` draws the pipeline. With `MLFLOW_TRACKING_URI` unset, runs go to a local `mlflow.db` instead of the stack.
- **New data:**
  1. Drop a batch CSV into `data/incoming/`.
  2. Within about 15 s the worker validates it, checks it for drift, and retrains if it drifted. A batch that fails validation is moved to `data/quarantine/`.
  3. The console's Pipeline and Data drift tabs show the run.

  Don't run `dvc` on the host while the worker runs: DVC locks the repo.
- **Admin calls** (rollback, starting a run) need the `X-API-Key` header with `SPATIFY_API_KEY` from `.env`.
- **Memory:** the stack uses about 2 GB, and every service has a hard limit (D30). The Ray stages use 4 CPUs and a 1 GB object store; on a bigger machine, raise `SPATIFY_RAY_CPUS` and `SPATIFY_RAY_OBJECT_STORE_MB`. `uv run python scripts/peak_memory.py -- <command>` reports a command's peak memory, and stops it if the machine runs low.
- **If the drive is remounted under a new path:** run `docker compose up -d --force-recreate` and `uv sync --all-extras --reinstall` (D1).

## Tests and checks

```bash
make help           # every target
make ci             # ruff, pytest, then dvc repro and the promotion gate (stack up)
make coverage       # the tests with a coverage report
make load-test      # /recommend latency at 1, 10 and 40 clients (stack up)
make failure-test   # kill a training run; the champion must keep serving (stack up)
make dashboards     # rebuild the Grafana dashboards from their specification
make api-docs       # refresh docs/openapi.json and the LLD's API reference from the code
```

172 automated tests; latest results in the [test report](docs/test_report.md).

## Layout

| Path | Contents |
|---|---|
| `src/spatify/` | The Python package: pipeline stages, model, gateway, worker, model-server supervisor, frontend |
| `services/` | One Dockerfile per Compose service |
| `monitoring/` | Prometheus (scrape config, recording and alert rules, their tests), Alertmanager, and Grafana (provisioning, dashboards built by `build_dashboards.py`) |
| `scripts/` | The simulator, load and failure tests, the bad-deployment demo, the docs export, setup helpers |
| `tests/` | The pytest suite |
| `data/` | DVC-tracked data (`raw/`, `base/`, `batches/`, `feedback/`); `incoming/` is the drop folder for new batches |
| `metrics/`, `reports/` | Each stage's metrics (DVC), the EDA, evaluation plots, benchmark and load-test results |
| `docs/` | The documents above and the course material |
| `evidence/`, `screenshots/` | Terminal logs, timings and screenshots behind every claim in the documents |
| `dvc.yaml`, `params.yaml`, `MLproject` | The pipeline, its settings, and the MLflow entry points |
