# Decision log

One entry per design decision: what was chosen, why, and what else was considered. These feed the HLD and the viva.

## D1. The repository lives in the course folder (`Project/spatify`) on the NTFS drive
- **Why:** it keeps the code next to other course materials and keeps my stuff organised.
- **Costs and mitigations:** NTFS (`ntfs3`) doesn't keep Unix file modes, so Git's `core.fileMode` is off and executable bits are set in the Dockerfiles rather than taken from the checkout. uv can't hardlink from its cache on ext4, so `link-mode = "copy"` is set. Service data (Postgres, MLflow, Prometheus, Grafana) lives in named Docker volumes, which sit on ext4.
- **Considered:** `~/spatify` on ext4, which avoids these workarounds.
- **Remounts:** after a reboot on 2026-11-15, the drive came back at `/run/media/athreya-ganesh/CE72C02072C00EE1` instead of `…EE11`. Containers keep the absolute bind-mount paths they were created with, so Docker mounted empty, root-owned folders at the old path: the worker crash-looped and the gateway stopped. The venv's scripts also start with absolute paths, so `uv run pytest` and `uv run dvc` stopped working. **Fix after a move:** `docker compose up -d --force-recreate` from the new path, and `uv sync --all-extras --reinstall`.
- **Note:** the planning TODO stays in `Project/`, outside the repo and out of version control.

## D2. uv instead of Conda/Mamba for the Python environment
- **Why:** `uv.lock` pins every package, including transitive ones, with hashes. One lock serves local development, the Docker images and the MLproject `python_env.yaml` (via `uv export`), and installs are fast.
- **Considered:** Conda/Mamba with `environment.yml`, as taught in L1. It works, but the exported file has no hashes and would be a second source of truth next to the Docker requirements.

## D3. Python 3.12
- **Why:** every dependency (Ray, MLflow, DVC, Streamlit) ships 3.12 wheels, and `python:3.12-slim` is a standard base image.
- **Considered:** the system Python 3.14. Ray 2.59 and MLflow 3.17 already run on 3.14 in my `~/aiops` venv, so moving up later is possible.

## D4. Dependencies split into extras per service
- **Why:** each Docker image installs only what it needs (`pipeline`, `serving` or `frontend`), which keeps images small, while one lock file keeps versions identical everywhere.
- **Pinned on purpose:** `ray==2.59.0` and `mlflow==3.17.1`, the versions the A3 code and the course exercises ran on. `uv.lock` pins the rest.

## D5. Dataset from the author's Hugging Face copy, tracked with `dvc import-url`
- **Why:** it is the same author and the same `dataset.csv` as the Kaggle listing, under a BSD licence, with no API token needed. `import-url` records the source URL in the `.dvc` file, so anyone without access to my DVC remote can rebuild the data with `dvc update` and `dvc repro`.
- **Considered:** the Kaggle CLI, which needs an account token.

## D6. DVC for data and models, Git LFS for binary documents
- **Why:** datasets, models and pipeline outputs change with the pipeline, so DVC versions them alongside `dvc.lock`. PDFs, screenshots and the demo video belong to the Git history, so they go in LFS (the L2 rule of thumb: large files that compress badly).
- **Guard:** a pre-commit hook rejects any other file over 500 KB.

## D7. A local folder as the DVC remote (`~/dvc-storage/spatify`)
- **Why:** cloud platforms are not allowed, and L2 uses a local remote.
- **Trade-off:** nobody else can `dvc pull` from it; D5 and the reproducible `make_batches` stage cover that.

## D8. Batch generator design
- **Base set:** 80% of the in-distribution tracks, split by `track_id`, so no track appears in both the base set and a batch.
- **normal batches:** unseen tracks from the base genres, for the no-drift case.
- **shifted batches:** 10 held-out genres chosen with the seed, for genre-mix drift. Tracks listed under a held-out genre are kept out of the base set entirely.
- **drifted batches:** unseen base-genre tracks with an additive shift (louder, more energetic, less acoustic). This stands in for the problem statement's "era" shift, because the dataset has no release year.
- **Planted errors:** 2% of each batch, half nulls and half out-of-range values, recorded in `manifest.json`, so the validate stage can be checked against known counts.
- **Considered:** sampling batch rows without the track-level split, which leaks base tracks into batches; a fixed genre list instead of a seeded draw (still possible: set `held_out_genres` to a list in `params.yaml`).

## D9. Ingest reads every column as text; validate does the typing
- **Why:** with typed reads, one malformed value (text in a number column) fails the whole file. Read as text, the bad row reaches validate and is rejected with a reason, and the rest of the batch survives.
- **Lineage:** every row keeps the name of the file it came from (`source`), so rejections and drift can be traced to a batch.

## D10. Validation policy: reject the row, warn, or stop the pipeline
- **Reject the row** (into `data/validated/rejected.csv`, with the reason and the offending value): an empty required column, a non-numeric value, or a value outside its range in `params.yaml`.
- **Warn and keep:** a missing artist, album or title, and zero tempo or time signature. In the raw data both are zero in 157 rows, 138 of them in "sleep" (119 of those rows are in the base set): silent or ambient tracks. They are real tracks, so rejecting them would bias the catalogue.
- **Stop the pipeline:** a missing column, or any source file with more than 5% of its rows rejected. The stage exits with an error, DVC stops, and the serving model is untouched (the problem statement's reliability requirement). The rate is per file, so a bad 2,000-row batch can't hide inside the 80,000-row base set, where it would be only 2.4% overall. Measured: a test batch with 20% of its rows corrupted had 431 of its 2,000 rows (21.6%) rejected, only 0.53% of all rows, and the stage stopped the pipeline (`evidence/phase1_bad_batch.log`).
- **Course link:** this follows the Great Expectations handout. The checks play the role of expectations, the validate stage is the checkpoint, and the metrics plus `rejected.csv` are the data docs; the hard-failure vs soft-warning split is the one from its credit-scoring case.
- **Considered:** Great Expectations or pandera. Either adds a dependency and runs outside Ray's batches, while these checks are a few lines of pandas inside `map_batches`. Revisit if the instructor confirms GE was taught (open question).

## D11. One row per track: tracks listed under several genres are merged
- **Why:** 16,299 tracks in the raw data appear under 2 to 9 genres (10,779 in the base set). As separate rows, the recommender could return the same track several times. Their audio features are identical and only popularity differs (720 tracks), so the merged row keeps the highest popularity and lists every genre in `genres`.
- **Effect:** in Phase 2, genre-match@K counts a match when the recommended track shares any genre with the seed.
- **How:** hash-partition by `track_id` (`repartition(keys=...)`), then merge each block with pandas. That's one Python call per block, where `groupby().map_groups()` (as in A3) makes one per track, about 64,000. The stage checks afterwards that no `track_id` appears twice.

## D12. Feature set and scaling
- **Features:** the nine continuous audio features, `mode`, and the key as a point on the circle of fifths (`key_sin`, `key_cos`), with a minor key placed on its relative major. As a plain number, key 11 (B) looks far from key 0 (C), and C and G (0 and 7) look further apart than C and C#; on the circle of fifths, keys that share most of their notes sit next to each other.
- **Scaling:** Ray's `StandardScaler` (L8 AIR notebook), fitted on the training set. Its means and standard deviations are saved to `data/features/scaler.json` and will be bundled with the model in Phase 2, so serving scales a query exactly as training did (no training-serving skew).
- **Drift reference:** a seeded sample of 10,000 tracks' unscaled features (`reference.parquet`), for PSI and KS checks in Phase 5.
- **Kept but not in the vector yet:** popularity, duration and explicit. Phase 2 can test adding them.

## D13. New batches are versioned with DVC (`data/incoming.dvc`)
- **Why:** the ingest stage depends on `data/incoming/`. If Git ignored the folder and DVC didn't track it, a fresh clone couldn't rebuild the ingested data. Now each batch drop is `dvc add data/incoming` plus a commit, so every model can be traced to the exact batches it saw (objective O4).

## D14. Throughput: Ray vs pandas
- **Measured** (`reports/benchmark.md`, 20 CPUs, 2026-11-14, against a `ray start --head` cluster): for 1, 10 and 20 copies of the base set (80,000, 800,000 and 1.6 million rows), Ray Data processed 14,702, 82,083 and 134,076 rows/s, and single-process pandas 56,821, 104,254 and 114,928 rows/s. Ray runs at 0.26x, 0.79x and 1.17x the speed of pandas, overtaking it between 10 and 20 copies.
- **Reading:** Ray pays fixed costs (task scheduling, a hash shuffle for the merge, moving data through the object store) that only pay off as the data grows. pandas runs in one process and must fit everything in that process's memory; Ray spreads work across cores and, with `RAY_ADDRESS`, across machines.
- **Kept Ray:** the course requires Ray or Spark for the data pipeline, and the same code scales out to a cluster without changes.

## D15. Ray stages run with `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`
- **Why:** when Ray finds `uv run` anywhere among a driver's parent processes, it builds a fresh environment for its worker processes from the project's base dependencies. That environment lacks the `pipeline` extra, so the workers fail with `No module named 'ray'` and the driver waits forever. `uv run pytest` and `uv run dvc repro` both hit this.
- **Fix:** turn that feature off; the workers then use the project's `.venv`, which already has every extra. The alternative is to activate `.venv` and run the commands without `uv run`.

## D16. Recommender: exact cosine nearest neighbours over weighted z-scores
- **Why:** the catalogue is 64,000 tracks with at most 12 features, so one matrix product scores a query exactly in 2–20 ms (`ms_per_query` in `metrics/evaluate.json`, and `evidence/phase2_serving_check.log`). An approximate index (FAISS, Annoy) would add a dependency for no gain at this size. Cosine over z-scores compares a track's profile relative to the average track.
- **Repeats of a song:** 8,360 tracks share artists and title with another track (2,993 songs released more than once, such as album and single versions). Results skip the seed's own song and any song already listed; otherwise the top result is often the seed again, and genre match would be inflated.
- **Considered:** scikit-learn `NearestNeighbors`, which gives the same neighbours but no control over skipping repeats.

## D17. Evaluation before feedback exists: genre match@10 on held-out seed tracks
- **Metric:** the share of the top 10 that shares at least one genre with the seed. With no feedback yet (Phase 5), genre is the available proxy for "sounds alike". Also reported: genre match@1, catalogue coverage (are a few "hub" tracks recommended to everyone?), mean similarity and time per query.
- **Split:** 10% of tracks are held out by a hash of `track_id`. Their genres are never used to fit a model; they stay in the catalogue, so they can still be recommended. A hash, unlike a random sample, keeps every track on the same side as new batches grow the catalogue.
- **Measured:** the best candidate reaches 13.5%, against 1.5% for random tracks (about 9 times better). That is low in absolute terms: audio features alone separate genres weakly. The neighbours of "Unholy" (dance, pop), for example, are tagged latin, turkish, chill, kids and electro. That is the cold-start limit the feedback loop in Phase 5 should improve on.
- **Considered:** leave-one-out over all tracks, which would leak the seed's genre into the Fisher weights.

## D18. Six candidates: three feature sets × two weightings, as nested runs
- **Feature sets:** all 12 features; `no_tonal` (without mode and key); `no_loudness` (without loudness, which moves with energy, r = 0.76). Dropping one group per run measures that group's importance (the guidelines' "track feature importance").
- **Weightings:** uniform, or Fisher: each feature weighted by between-genre over within-genre variance, fitted on the training split only.
- **Measured (genre match@10):** all-fisher 13.48%, no_tonal-uniform 13.47%, no_tonal-fisher 13.19%, all-uniform 12.82%, no_loudness-fisher 12.33%, no_loudness-uniform 11.82%. A difference between two candidates on the same 2,000 seeds has a standard error of about 0.25 points.
- **Reading:** loudness carries genre information despite its correlation with energy (dropping it costs about 1 point), so the EDA's hint was wrong. Key and mode add noise under uniform weights, and dropping them gains 0.65 points. Fisher weighting finds the same thing by itself: it gives key 0.21–0.25 and mode 0.38, against about 1.3 for speechiness, instrumentalness and acousticness. That is why all-fisher and no_tonal-uniform tie (difference 0.01 points).

## D19. Unset sliders sit at the mean (z = 0), not the median
- **Why:** the TODO planned the median, and a unit test caught the problem. The median of `mode` is 1 (major), so a query that set only energy and danceability was pulled toward major-key tracks, and a middling jazz track outranked a close dance match. With cosine over z-scores, the mean is the neutral value: an unset feature adds no direction to the query.
- **Side effect:** tracks with extreme values on unset features score slightly lower, a mild preference for typical values on what the user didn't specify.
- **Empty queries:** a request with no seed track and no slider the model uses is rejected with HTTP 400.

## D20. Packaging: a custom pyfunc that carries its catalogue, scaler and code (L10)
- **Inside the model:** `catalogue.parquet`; `model_info.json` with the feature columns, the features stage's scaler (so serving scales a query exactly as training did), the weights, the default K and the training run's ID; and the `spatify` package (`code_paths`).
- **Signature:** every input optional (`track_id` as a string, sliders as doubles so JSON integers are accepted), K as an inference parameter (default 10). Every answer carries `model_run_id`. A model can't know its registry version when it is logged, so the registry maps run to version.
- **Requirements:** only mlflow, numpy, pandas and pyarrow, pinned to the installed versions, which come from `uv.lock`. MLflow 3.17 would otherwise copy the whole `uv.lock` into every model (`MLFLOW_UV_AUTO_DETECT`), a second, conflicting environment spec: the project's base packages, without mlflow.
- **Errors:** an unknown track or an empty query raises an MLflow `INVALID_PARAMETER_VALUE` error, which the server returns as HTTP 400; a wrongly typed input fails schema enforcement, also 400 (`evidence/phase2_serving_check.log`).
- **Considered:** MLflow's newer "models from code" style; kept the course's `PythonModel` pattern.

## D21. Model stages run as MLflow Projects; data stages don't
- **How:** DVC runs `mlflow run . -e <stage> --env-manager=local --experiment-name spatify` for train, evaluate and register; the scripts never call `set_experiment()` (the L3 iris gotcha). Each pipeline run leaves a train run (a parent and 6 nested candidates), an evaluate run and a register run, linked by `train_run_id` tags.
- **Data stages:** ingest, validate and features stay plain Python under DVC, since they log nothing to MLflow. Their metrics are attached to the train run (`data_metrics/`), so a model links to the quality of the data it saw.
- **Logged beyond autolog:** git commit and whether the tree was dirty, DVC data version, trigger, environment, Fisher weights and their figure, training feature distributions and the drift-reference sample, `params.yaml` and `python_env.yaml`.
- **Environment:** `python_env.yaml` is generated from `uv.lock` by the `python_env` DVC stage, so it changes only when the lock does. The stages run with `--env-manager=local` inside the uv environment, as the course recommends. The file doesn't install the `spatify` package itself, so another environment manager would also need `pip install -e .`.
- **Tracking store:** MLflow 3.17's default, `sqlite:///mlflow.db` in the repo root (ignored by Git), with artifacts in `mlruns/`. Phase 3 moves this to the MLflow server in Compose, backed by Postgres.

## D22. Promotion gate and registry aliases (L10, Task A)
- **Gate:** the best candidate (by `mlflow.search_runs` on genre match@10) is registered as a new version under the `challenger` alias. It's tagged with dataset version, git commit, trigger, variant and training run. The challenger and the champion are then scored on the same 2,000 held-out seeds (only seeds both catalogues contain), and the challenger is promoted if it wins by at least 0.005.
- **Why 0.005:** the standard error of a difference between two models on the same seeds is about 0.0025, so the bar is two standard errors. A smaller gain could be noise, and swapping models on noise only churns the version that feedback is attributed to.
- **Aliases:** `champion` is served; `previous` is the rollback target; `challenger` is the latest candidate that wasn't promoted. Stages are deprecated (L3, L10).
- **Not promoting is not a failure:** the stage exits 0 and the champion keeps serving.
- **Measured:** the first run registered version 1 and promoted it. Retraining on unchanged data registered version 2 with a gain of 0.0, so it was not promoted (`evidence/phase2_gate_not_promoted.log`). The two runs' metrics are identical apart from timings.

## D23. The serving stack: four Compose services
- **Services:** `postgres` (the feedback store, plus a second database for MLflow), `mlflow` (tracking server and model registry), `model-server` and `gateway`. The frontend (Phase 4), worker (Phase 5) and Prometheus and Grafana (Phase 6) join later.
- **Addresses:** services reach each other by name (`mlflow:5000`, `model-server:8080`, `postgres:5432`), set through environment variables. Only the gateway (8000) and the MLflow UI (5000) are published, and only on 127.0.0.1; Postgres and the model server stay on the Compose network.
- **Secrets:** in `.env`, which Git ignores. `scripts/make_env.py` fills it with random values and `.env.example` documents it. Compose refuses to start if a secret is missing.
- **Start order:** health checks, with `depends_on: condition: service_healthy` for Postgres and MLflow. The gateway waits for the model server only to *start*: on a fresh stack there is no champion yet, so the model server stays unhealthy, and the gateway should still come up and report it (`/health` 503).
- **First champion:** the Phase 2 registry in the local `mlflow.db` isn't migrated. The stack's first champion comes from one pipeline run against the server (`MLFLOW_TRACKING_URI=http://localhost:5000 dvc repro --force --downstream train`, in `evidence/phase3_bootstrap.log`). It scored the same 0.1348 as in Phase 2.

## D24. Model server: a supervisor that follows the champion alias
- **Why:** `mlflow models serve -m models:/spatify@champion` resolves the alias once, at start-up (L10), so promotions and rollbacks would never reach a running server. The supervisor (`src/spatify/model_server.py`) resolves the alias to a version number itself and serves that exact version (`models:/spatify/<n>`). It re-reads the alias every 10 s and restarts the server when it moves. It also restarts a crashed server, and keeps serving if the registry is unreachable.
- **Measured** (`evidence/phase3_rollback.log`): after a rollback through the gateway, answers came from the new version 9.5–13.4 s later. Up to 10 s of that is the poll interval, during which the old version still answers normally. About 5 s is the restart, during which `/recommend` answers 503 "try again in a few seconds".
- **Considered:** `mlflow models build-docker`, which bakes one version into an image, so every promotion or rollback would need an image build and a redeploy. Also two servers with a switch in front (blue-green), which would remove the 5 s gap; not worth the extra moving parts here.

## D25. MLflow server: Postgres backend, artifacts served by the server
- **How:** runs and the registry live in Postgres (`postgresql+psycopg://…/mlflow`). Artifacts live in the `mlflow-artifacts` volume and pass through the server (`--serve-artifacts`), so the host pipeline, the model server and the gateway need only HTTP to MLflow, no shared filesystem.
- **`--allowed-hosts`:** MLflow 3 rejects Host headers other than localhost and private IPs, a protection against DNS rebinding. Containers call it as `mlflow:5000`, so that name is allowed explicitly.

## D26. Gateway: FastAPI, and every served list is stored
- **Boundary:** the frontend reaches the model, the registry and the database only through the gateway's REST API (`/docs`).
- **Traceable feedback:** every `/recommend` answer is saved (request ID, seed, sliders, K, model version, track IDs). Feedback must name a request ID and a track in that list, so each like or skip carries the model version and the position it was shown at (FR2), and feedback on tracks that weren't shown is refused. Phase 5's hit rate, like rate per version and exploration logging build on this. The model returns its training run ID; the gateway maps it to a version through the registry.
- **`/health`:** 200 only when Postgres and the model server answer (the TA guide deducts for a health check that always returns 200). MLflow is reported but not required, since serving continues while MLflow is down.
- **Errors:** a request that fails validation gets a 422 naming the field, before it reaches the model. An unknown seed gets a 404 pointing to search; the model server being down gets a 503 with a retry message, as does the database being down. An impossible rollback gets a 409, and anything unexpected a 500 with the request ID. Every response carries `X-Request-ID`, and every request is logged as one JSON line (`evidence/phase3_stack.txt`).
- **Search** runs over the champion's own catalogue, downloaded from its model files when the champion changes, so a seed found by search is always one the served model knows.
- **mlflow-skinny,** the client without the server, keeps the gateway image smaller.

## D27. Images: multi-stage builds from the same lock file
- **How:** the builder stage is `python:3.12.15-slim` plus uv 0.13.5. `uv sync --frozen` installs the dependencies first (a cached layer that survives code changes), then the code as a regular package. The final stage is the slim image plus the virtual environment only, running as a non-root user. `.dockerignore` is an allow-list: the build context is just `pyproject.toml`, `uv.lock`, `.python-version`, `README.md` and `src/`.
- **One extra per image:** the gateway installs `serving`; MLflow and the model server install `mlflow-server`. scikit-learn and scipy moved from the base dependencies to `pipeline`, since nothing outside the pipeline uses them; `python_env.yaml` came out identical.
- **Sizes:** gateway 801 MB; MLflow and model server 1.29 GB each, sharing almost all their layers. Base images are pinned (`python:3.12.15-slim`, `postgres:17.12-alpine`, `uv:0.13.5`).

## D28. Admin endpoints need a hashed API key
- **How:** `POST /models/rollback` and `POST /pipeline/trigger` need an `X-API-Key` header. The gateway holds only the key's SHA-256 (`API_KEY_SHA256`), compares in constant time, and refuses admin actions altogether if no hash is set. The plain key stays in `.env` for the operator and, in Phase 4, the frontend.

## D29. Pipeline triggers go into a queue table in Postgres
- **How:** `POST /pipeline/trigger` inserts a `requested` row in `pipeline_runs`. The worker (Phase 5) claims requested rows, runs the pipeline and fills in status, stages, durations and the model version. The gateway and the worker share Postgres rather than calling each other over HTTP, so a trigger survives a worker restart, and the run history the console shows comes from the same table.

## D30. A memory budget for a 16 GB laptop
- **Why:** the laptop has 15 GB, and a browser and VS Code take about 6 GB of it. With the Compose stack at about 2.4 GB and an uncapped Ray pipeline on top, the machine was deep in swap (2.7 of 4 GB in use) and at risk of freezing.
- **MLflow server:** `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` turns off MLflow 3's job runner, 7 background processes of about 150 MB each, which serve GenAI features (scorers, prompt optimisation, trace roll-ups) that Spatify doesn't use. It also runs one uvicorn worker instead of two.
- **Model server:** one worker instead of two. Each worker holds its own copy of the model (about 280 MB), and one answers in a few milliseconds. The supervisor reads the alias with one REST call instead of importing the mlflow package, which took it from 160 MB to 27 MB.
- **Hard limits:** every service has a memory limit with no swap: Postgres 256 MB, MLflow 768 MB, the model server 768 MB and the gateway 512 MB. Each is 1.5–2× the measured use, so a runaway service is killed and restarted by Compose instead of pushing the laptop into swap.
- **Ray:** a local Ray instance is capped at 4 CPUs and a 1 GB object store (`SPATIFY_RAY_CPUS`, `SPATIFY_RAY_OBJECT_STORE_MB`). By default Ray starts a worker per CPU (20 here) and reserves 30% of RAM. It starts with `address="local"`, so a record of a dead cluster in `/tmp/ray` can't make a stage try to join it, which had happened after an interrupted `ray start --head`.
- **Measured:**
  - The stack went from about 2.4 GB to 1.17 GB (`docker stats`).
  - The whole pipeline (`dvc repro`, every stage, with the stack running) raised used memory by at most 1.9 GB and left at least 4.5 GB available, in 56 s (`evidence/phase3_capped_pipeline_memory.log`). The results were unchanged: genre match@10 was 0.1348 again.
  - The features stage took 5.3 s on 4 CPUs against 7.6 s on 20. It produced the same tracks and drift reference, with scaled values within 2e-15.
- **The benchmark:** the numbers in D14 were measured on 20 CPUs, up to 20 copies. Capped at 4 CPUs and run at 10 copies, it raised used memory by at most 2.4 GB, and Ray reached 65,674 rows/s against pandas' 94,504 (0.69×): Ray's advantage needs both cores and data.
- **Considered:** tuning Postgres (it uses 13 MB of program memory, so there was nothing to gain); running uvicorn directly instead of `mlflow models serve`, which would save about 240 MB but depend on MLflow's internals.

## D31. Rerunning on the same data gives the same DVC hash
- **Problem:** DVC hashes a folder from its file names and their contents. Ray named every Parquet file after a random ID chosen per write, so each rerun produced new names. Ray also adds up the scaler's partial sums in whatever order its tasks finish, so the last digits of the means and standard deviations varied too. Identical data therefore got a new hash every run. That changed every model's `dataset_version` tag and made DVC retrain after a rerun that changed nothing; the reproducibility drill showed it.
- **Fix:** the Ray stages write through `write_parquet` in `src/spatify/pipeline/common.py`. It names files by position (`part-00003-0.parquet`) and writes in OVERWRITE mode, since Ray insists on the random ID only when appending to a folder, and the stages always write into a folder they have just emptied. The features stage also rounds the fitted scaler statistics to 12 significant digits before transforming, an error of at most 1e-12 relative.
- **Measured** (`evidence/phase1_deterministic_outputs.log`):
  - Two runs of ingest, validate and features on the same data gave byte-identical files (13, 14 and 6 of them).
  - After a forced rerun of those stages, the hashes in `dvc.lock` were unchanged, and `dvc repro` then skipped train, evaluate and register.
  - `tests/test_pipeline.py` checks the same thing on a small catalogue.
- **Limit:** file boundaries follow Ray's blocks, and the number of blocks depends on the CPU cap (the features stage uses one block per CPU, D30). So a run with a different `SPATIFY_RAY_CPUS` gets a different hash for the same data; reruns with the same settings get the same one.
- **Considered:** writing one file per stage (`repartition(1)`). That would make the hash independent of the CPU count, but it funnels all the data through a single task, against the point of a Ray pipeline.

## D32. The frontend: Streamlit, talking only to the gateway
- **Pages:** *Discover* for listeners, the *Operator console*, and *Help*, the in-app user manual. Streamlit, as in A4 and the problem statement, served from its own container on 8501. It reaches the backend only through the gateway's REST API, at `API_BASE_URL` (`spatify.frontend.api`). Its image has neither MLflow nor FastAPI, so it couldn't take a shortcut if it tried.
- **Discover:**
  - Start from a song (search, then pick), from sliders, or both.
  - Each slider has a switch, so an unset feature stays unset (D19), and the six intuitive features come first with the rest under *More options*. The button stays disabled until a song or a slider is chosen.
  - Each suggestion shows its similarity, its "why" labels (D33), and Like and Skip buttons. These send `/feedback` and grey out once used, so a click can't be sent twice.
  - The list says which model version chose it.
- **Foolproofing:** every gateway failure becomes a message saying what to do ("try again in a few seconds", "check the spelling"), never a stack trace. Song titles are escaped, so Markdown characters in them can't break the page. Search requires two letters.
- **The admin key:** rollback and pipeline runs need it, and the operator types it into the console's sidebar. It stays in that browser session; the frontend's container never holds it. D28 had planned to give the container the key, but then anyone who could open the console could roll back models.
- **Metrics:** prometheus_client serves `/metrics` on its own port, 9101, as in A4, using A4's guard against Streamlit re-importing modules. They count page views, gateway calls by endpoint and outcome, gateway latency as the user sees it, like and skip clicks, and the error messages shown.
- **Charts:** they follow A4's "7 Commandments": a title, a one-line description, axes named with units and scale, a legend whenever colour means something, and the Okabe-Ito palette.
- **Tests:** Streamlit's AppTest runs every page headlessly against a fake gateway and checks both what the page shows and what it sends (`tests/test_frontend.py`).

## D33. "Why" labels: each feature's exact share of the score
- **How:** the score is a cosine similarity, a sum over features of (query × track), both as weighted z-scores of length 1. So each feature's product is its exact share of the score. The labels name the features with the largest positive shares, at most three and at least 5% of the score each. Those are the features where the query and the track are both well above, or both well below, the average track ("energetic", "acoustic", "minor key"). A slider-only query is explained only by the sliders set, since unset features sit at the mean and contribute nothing.
- **Where:** the gateway computes them with the champion's own scaler, weights and catalogue, read from the files packaged in the model version (`spatify.model.explain`). The labels therefore describe the model that answered. If another model answered (the champion has just changed and the gateway hasn't reloaded yet), the gateway gives no labels rather than wrong ones.
- **Considered:** returning the labels from the model server, which would change the served model and its DVC stage (and so retrain it); or labels based only on closeness, which can name features that barely affect the ranking.

## D34. The console's pipeline view comes from DVC's own files
- **How:** the gateway mounts `dvc.yaml`, `dvc.lock` and `metrics/` read-only, and serves `/pipeline/dag`:
  - **Stages:** in `dvc.yaml` order.
  - **Edges:** wherever one stage's output (or metrics file) is another's input.
  - **Inputs:** the data that no stage produces (the raw CSV and the incoming-batch folder).
  - **Throughput:** each stage's rows and rows per second, from its metrics file.
  The diagram matches `dvc dag`.
- **Colours:** a stage's last result comes from the latest run the worker recorded, once Phase 5's worker reports results per stage. Until then a stage recorded in `dvc.lock` counts as last run successfully, and one missing from it as never run.
- **Considered:** running `dvc dag --dot` inside the gateway, which would put DVC and the whole repo into the gateway image.

## D35. Drift: PSI decides, KS reports, against the base catalogue
- **How:** the `drift_check` stage compares each incoming batch with the reference, feature by feature, for the nine continuous audio features:
  - **PSI** over ten bins cut at the reference's deciles.
  - **The two-sample KS test** (statistic and p-value).
  The reference is the base catalogue's validated rows, the data every model is trained on.
- **Rule:** a feature is *major* at PSI ≥ 0.2, *moderate* at 0.1–0.2 and *stable* below, the usual rule of thumb. A batch has drifted when any feature is major, or at least two are moderate (`params.yaml` → `drift`).
- **Calibrated on the generated batches:**
  - **Normal batches:** the largest PSI is 0.005–0.012.
  - **Shifted batches** (10 held-out genres): energy reaches 0.205–0.212, and two or three more features are moderate.
  - **Drifted batches** (energy, loudness and acousticness shifted): acousticness reaches about 1.6.

  A single 0.2 cut-off would catch the shifted batches by a hair; the two-tier rule catches them clearly, and normal batches sit far below either line.
- **Why KS doesn't decide:** against about 80,000 reference rows, KS finds even normal batches different (normal_03 has 2 features at p < 0.01). Its p-values measure sample size as much as shift. It is reported next to PSI.
- **Where it lives:**
  - It sits after validate and beside features, so `dvc repro drift_check` checks new data without retraining.
  - The results are in `metrics/drift.json` (DVC metrics), the gateway's `/drift` (the console), and the worker's `spatify_drift_psi{feature}` and `spatify_drift_detected` gauges (Prometheus, for Phase 6's alerts). Exposing drift scores as metrics is the problem statement's FR4.
- **Limit:** after a drift-triggered retrain, the model has learnt the shifted batch, but later batches are still compared with the base catalogue, so a similar batch is flagged again. Re-baselining on the champion's own training data (logged as `drift/reference.parquet` with each training run) would tie a plain DVC stage to MLflow; left as it is.

## D36. The worker: a polling loop that runs DVC in the mounted repo
- **Each check (every 15 s):**
  1. **A run queued from the console or the API:** bring the data up to date, then retrain. If training was already up to date, the worker forces it, since someone asked for a retrain.
  2. **Otherwise, new, changed or removed files in `data/incoming`:** `dvc add data/incoming`, then `dvc repro drift_check`. If a new batch has drifted, retrain with the trigger `drift`. If not, record the data and leave the model alone, as FR5 asks ("retrain when drift or feedback crosses a threshold").
  3. **Otherwise, nothing.**
- **What's recorded:**
  - **`pipeline_runs`:** each run's trigger and result, each stage's result and seconds (read from DVC's own output), the model version registered, whether it was promoted, and a one-line explanation. The console's run history, the pipeline diagram's colours and the worker's Prometheus metrics all come from this.
  - **`incoming_batches`:** every batch file by its SHA-256, and what became of it.
- **Failure leaves the champion serving:** only the register stage moves the alias, and DVC never runs a stage after a failed one.
  - **A batch that validation rejects** is moved to `data/quarantine/`; otherwise it would fail again on every check. Good batches dropped with it go through on the next check.
  - **A failure anywhere else** marks that run's batches as failed, so the worker doesn't retry them every 15 s.
  - **A run left 'running' by a worker that stopped mid-run** is marked failed when the worker restarts. Compose gives a stopping worker 3 minutes to finish its run.
  - **No retry storms.** An unexpected error marks the run's batches failed too, and after any failed run the worker waits a full poll interval. The first version had neither: a missing user account made it fail and retry about once a second (236 runs). Restarting the worker puts failed batches back in line, so after fixing a cause, a restart retries.
- **A batch dropped in again after removal counts as new** (fixed in step 6.3). The first version matched files by name and SHA-256 and skipped a returning file whose record said `removed`, so re-running a batch did nothing.
- **It runs the repo's own code** (`PYTHONPATH=/repo/src`; the image holds only the dependencies). So the stages that run are exactly the code DVC hashed as their dependencies, even after an edit without a rebuild.
- **It runs as the repo's owner** (uid 1000), since the NTFS mount lets only that user write. The image has an account with that uid (a build argument): DVC and MLflow look up the current user's name, and with only a bare uid the first version failed with `getpwuid(): uid not found`.
- **Memory:** a 3 GB limit, and 1.5 GB of `/dev/shm` for Ray's 1 GB object store. Ray is capped at 4 CPUs (D30).
- **It doesn't commit.** It leaves `dvc.lock` and `data/incoming.dvc` updated in the working tree for the operator to commit; MLflow tags the models it trains `git_dirty=true`. Committing from a container into the user's repo would skip the pre-commit hooks and mix with whatever else the user has in progress.
- **One pipeline at a time:** DVC locks the repo, so don't run `dvc` on the host while the worker is running.
- **Considered:** watching the folder with inotify (unreliable on bind mounts and NTFS, hence polling); a fixed schedule (a polled queue starts a requested run within seconds). Phase 6's alerts can queue runs through the same table.

## D37. Learning from likes and skips: feature weights from a logistic regression
- **What it learns:** new feature weights for the same cosine recommender. A logistic regression predicts a like from the squared difference between the query and the rated track, feature by feature, in z-scores. A feature where closeness raises the chance of a like gets weight sqrt(−coefficient); one where it doesn't gets 0. The weights are scaled to a mean square of 1, like the Fisher weights (D18). Balanced classes, C = 1, every feature.
  - **Z-scores, not the recommender's unit vectors:** unit vectors tie every feature to all the others; z-scores recovered the simulated listeners' taste better.
  - **Why this model:** the served model stays one kind of model (nearest neighbours in one pyfunc), and the weights explain themselves (the "why" labels, D33, and a weights plot in MLflow). The problem statement asks for "a re-ranking or retraining step"; this is the retraining kind.
  - **Considered:** a separate re-ranker on top of the recommender (two models to serve and version together); collaborative filtering (needs listening histories, which cold-start listeners don't have, and is out of scope).
- **Where it enters the pipeline:**
  - **The snapshot:** before every retrain, the worker exports all likes and skips from Postgres to `data/feedback/feedback.parquet` and runs `dvc add data/feedback`. Train, evaluate and register depend on it, so every model traces to the exact ratings it learnt from.
  - **Train** adds an `all-feedback` candidate when the earlier 80% of lists (by time) hold at least 1,000 usable ratings with both likes and skips; otherwise it logs why there is none.
  - **The trigger:** the worker retrains (trigger `feedback`) once 2,000 ratings have arrived since the last snapshot (`feedback.retrain_every`).
- **How it's judged:** on the latest 20% of lists, each list's rated tracks are re-ranked by the model's similarity to the query, with likes as the relevant items: NDCG@3 and hit rate@3, against a random order. Once there are at least 30 held-out lists, NDCG@3 picks the best candidate and decides promotion (minimum gain 0.01); before that, genre match@10 does, as in D22. Feedback measures what listeners want; genre only stood in for it.
- **Measured offline** (`scripts/feedback_experiment.py`; 3 seeds; listeners with the hidden taste of D38):
  - **Against a uniform-weight champion** (`evidence/phase5_feedback_offline.log`, 2,000 ratings): NDCG@3 0.58 / 0.36 / 0.62 → 0.67 / 0.55 / 0.66, and the like rate 37–42% → 47–55%.
  - **Against a Fisher-weight champion, like version 1** (`evidence/phase5_feedback_offline_fisher.log`, 4,000 ratings): the like rate still rises (44–49% → 49–55%), but NDCG@3 on the champion's held-out lists is mixed (0.576 → 0.553, 0.569 → 0.572, 0.583 → 0.629). The gate would promote in one seed of three.
  - **The learnt weights** put energy, valence, acousticness and danceability on top in every seed. Those are the four features the simulated listeners care about most (D38), and the system never sees that list.
- **The real run (step 5.4, `evidence/phase5_worker_feedback.log`):** with 3,955 ratings, `all-feedback` scored 0.577 on 72 held-out lists against 0.483 for `all-uniform`. The Fisher champion scored 0.587, so the best candidate was another `all-fisher` (gain 0.0) and nothing was promoted.
- **The second real run (2026-11-18, during step 6.3's load):** with 5,955 ratings, `all-feedback` scored 0.584 against the champion's 0.498 on 84 held-out lists and was promoted (version 8). It won on version 1's own lists too (0.585 against 0.487 on 72 of them), so the gate's bias didn't decide it. It had half as many ratings again to learn from, and on about 80 lists the comparison varies from run to run. Retrain to redeploy took 76.5 s; the model server was answering 4.8 s after it started version 8. Listeners got the gateway's friendly 503 for about 11 s (795 requests), too short for `GatewayErrorsHigh`, which needs 2 minutes.
- **Why the gate and the like rate disagree:** the held-out lists are the champion's own picks, ranked by its own weights. Re-ranking them favours the model that chose them (off-policy evaluation). The like rate on a model's own lists is the fair test, but it needs live traffic: shadow or canary serving with the `challenger` alias (L6, Stretch). Until then the gate is conservative. It can keep a better model out, but it never lets a worse one in.
- **Exploration** (the hidden feedback loop from L1): in a list of 10, the lowest slot goes to a random track (`EXPLORATION_RATE` 0.1; lists under 10 get none). It is stored in `recommendations.explored`, carried into the snapshot, and shown as a "wildcard" on Discover. Without it, feedback covers only tracks the model already ranks high, so it can't learn where its ranking is wrong. **Cost:** random picks are liked about 6% of the time, against about 42% for the model's.
- **Feedback drift:** the like rate of the latest 200 ratings against the 1,000 before them. It's drift at a drop of 0.15 or more, once there are 200 recent and 500 earlier ratings. It shows on `/drift`, the console and the gauges `spatify_feedback_like_rate{window}` and `spatify_feedback_drift`. A falling like rate while PSI stays stable is L1's concept-drift signal.
  - **It doesn't retrain by itself:** right after a promotion, a falling like rate more likely means a bad model than new taste, and retraining on that model's feedback is slower than rolling back. An operator decides (D39).
- **Retrain-to-redeploy time:** after a promotion, the worker asks the model server every second until it answers with the new model's run ID. That wait is recorded as the run's `serving` stage, and the time from the run's start goes to `spatify_retrain_to_redeploy_seconds`.
- **Limits:**
  - The listeners are simulated (D38): the gain shows the loop works, not how much real listeners would gain.
  - **Off-policy evaluation** favours the champion (see above). The like rate per version (console, Models) is the on-policy check.
  - Candidates that drop features skip slider lists they can't read, so they're judged on slightly different lists.

## D38. Simulated listeners, with hidden taste
- **Why:** real listeners are few. The problem statement says feedback "will be partly produced by a documented user-simulation script", and the report must say so.
- **A listener** (`spatify.simulation`):
  - **Taste:** they care about energy (importance 3), valence (3), danceability (2), acousticness (2) and tempo (1). Each listener's importances vary (log-normal, σ 0.3).
  - **Tolerance:** between 0.04 and 0.09.
  - **Ratings:** they like a track with a logistic probability that falls as the importance-weighted mean gap from what they want grows (slope 25; tempo gaps divided by 200). Even an identical song isn't a sure like, so the feedback is noisy.
- **Each list is a new listener**, a cold-start user. 70% start from a random seed song and want songs that sound like it; 30% set two or three sliders and care only about those.
- **Calibration:** the cold-start model's picks are liked about 40–43% of the time (seed lists about 44%, slider lists about 22%) and random picks about 6% (`evidence/phase5_feedback_offline.log`).
- **How it runs:**
  - **`scripts/simulate_users.py`** goes through the real gateway (`/recommend`, then `/feedback` per track) with a thread pool, so it doubles as the load generator for Phase 6. Each list has its own seeded random generator, so thread timing doesn't change a run; the seed is printed.
  - **Its "ears"** are the local catalogue (`data/features/tracks`), because the API returns no audio features.
  - **`offline_lists`** runs the same listeners against a model directly, for the tests and `scripts/feedback_experiment.py`.
- **Limits:** real taste isn't five fixed features. These listeners ignore genre, artist and popularity, and the like-rate level is a calibration choice. The report states this.

## D39. A forced bad deployment, for the rollback demo
- **What:** `scripts/force_bad_model.py` copies the champion with each track's audio features taken from another random track, as a bad join would do. It registers the copy with the trigger `forced_bad` and makes it champion directly, skipping the promotion gate; the old champion becomes `previous`. It needs `--yes`.
- **Why this kind of break:** it passes every check before serving (same columns, ranges and signature) and fails only where listeners notice. That is the kind of failure monitoring is for, and the problem statement's evaluation plan asks for "successful rollback after a forced bad deployment". Offline, listeners like about 6% of its picks, against about 41% for the model it copied.
- **What follows:**
  1. The model server switches within about 15 s (D24).
  2. The like rate falls (console, Models), and feedback drift fires (console, Data drift; D37).
  3. An operator rolls back from the console or `POST /models/rollback`.
  4. The broken version stays registered, tagged, for the record.
- **If nobody rolls back:** after 2,000 more ratings, the worker's feedback trigger retrains, and a challenger will most likely beat the broken champion at the gate. That is a slower, automatic way out.
- **Why a script, not a console button:** a button that deploys a broken model has no place in a real operator console.

## D40. Monitoring: Prometheus scrapes every component on the Compose network
- **What:** Prometheus (5-second scrapes, 2 days of data) and two exporters join the Compose stack. Every target is a service name, not `localhost` as in A4's host networking. Each one:

  | Job | What it reports |
  |---|---|
  | `gateway` | requests, latency (histograms), model-server calls, feedback-store queries, feedback, like rates, alerts received |
  | `frontend` | page views, gateway calls by outcome, errors shown to listeners |
  | `worker` | pipeline runs, stage times, data-stage throughput, PSI, pending feedback, retrain-to-redeploy |
  | `model-server` | the supervisor: version served, restarts by reason, switch time, memory of the serving processes |
  | `mlflow` | the tracking server's requests (`mlflow server --expose-prometheus`) |
  | `postgres` | postgres-exporter: up, rows per table, database size |
  | `node` | node-exporter: the laptop's CPU, memory and disk (A4's service and relabel rules) |
  | `prometheus`, `alertmanager`, `grafana` | the monitoring stack itself |

- **Metrics added for this phase:**
  - **The model server's supervisor.** `mlflow models serve` has no `/metrics`, so the supervisor exports the server's side. The gateway already measures the requests. The memory gauge adds up the whole process tree from `/proc`, because the model lives in a child process.
  - **The worker** turns each data stage's rows per second from `metrics/<stage>.json` into gauges, so Ray Data throughput is on a dashboard, not only in DVC.
  - **The gateway:**
    - a **Summary** of the list position of every liked and skipped song. It completes L9's metric types: Counter, Gauge, Histogram, Summary and Info. Its average shows whether liked songs sit near the top.
    - a counter of alerts received.
  - **MLflow:** `prometheus-flask-exporter` joined the `mlflow-server` extra. `--expose-prometheus` was checked on MLflow 3.17.1 against a throwaway server before relying on it.
- **The frontend's metrics start with its process** (found in step 6.3). Streamlit runs `app.py` only when a browser opens the app, so a metrics server started there left `frontend:9101` closed after every restart until someone visited. `ServiceDown` fired for the frontend after each rebuild. `python -m spatify.frontend` now starts it before Streamlit.
- **Feedback store:** postgres-exporter, rather than only the gateway's view of its own queries, so the store itself reports (the problem statement's "all major components emit metrics").
- **Memory:** limits of Prometheus 384 MB, Grafana 512 MB (it used about 320 MB while starting), Alertmanager, node-exporter and postgres-exporter 64 MB each. Together they use about 0.5 GB idle.
- **Considered:**
  - The Pushgateway: every Spatify process is long-running and can be scraped. The worker's pipeline stages are short, but their results reach the worker's own gauges.
  - cAdvisor for per-container memory: about 100 MB more, and the supervisor and process gauges cover the services that matter.
  - Scraping Ray's own metrics: Ray runs only inside pipeline stages, so there's nothing to scrape between runs.

## D41. Alerts: rules in Prometheus, delivered by Alertmanager to the gateway
- **Recording rules:**
  - p95 `/recommend` latency, the 5xx share, requests per endpoint, and the like rate per model version.
  - The like-rate rule gives 0%, not nothing, for a version with skips but no likes (a broken model must show up).
  - The alerts and dashboards share these rules, so both read the same numbers.
- **Alert rules:**

  | Alert | Fires when |
  |---|---|
  | `ServiceDown` | a target hasn't been scraped for 1 minute |
  | `ModelServerUnreachable` | the gateway's probe of the model server has failed for 1 minute |
  | `RecommendLatencyHigh` | p95 `/recommend` latency is above 300 ms for 2 minutes (the problem statement's target) |
  | `GatewayErrorsHigh` | more than 5% of requests get a 5xx answer |
  | `DataDrift` | the worker's verdict on the latest batch is "drifted" (D35) |
  | `FeedbackDrift` | the gateway's feedback-drift flag is on (D37) |
  | `PipelineRunFailed` | a run failed in the last 15 minutes |
  | `HostLowMemory` | less than 10% of memory is available (from A4) |

  **Tested** with promtool unit tests (`monitoring/prometheus/tests/rules_test.yml`): synthetic series in, and the exact alerts and their text out.
- **Delivery:** Alertmanager groups alerts by name and posts firing and resolved alerts to the gateway's `POST /alerts`. The gateway stores them in Postgres, and the console's Monitoring tab shows the firing ones first.
  - **Why the gateway:** the lecture's sample posts to webhook.site, an outside service. A local receiver keeps alerts on the laptop and puts them where operators already look.
  - **Authentication:** the endpoint wants the admin key as a Bearer token. Alertmanager's container writes it from `.env` to a file only it can read, so no secret is in the committed config.
- **Alerts don't act:**
  - The worker already retrains on drift by itself (D36), so a `DataDrift` webhook that queued a retrain would start the same retrain twice.
  - `FeedbackDrift` is for an operator to judge (D37, D39).
- **A critical alert hides the warnings** it causes on the same job (an inhibit rule). If the gateway is down, its latency and error alerts would only repeat that.

## D42. Dashboards: generated from one spec, embedded in the console
- **Four dashboards**, provisioned from files into a Grafana folder "Spatify":
  - **Service health:** what's up, traffic, latency, errors, the feedback store, MLflow and the frontend.
  - **Model and feedback:** the version served and its switches, like rates by version and over time, the position of liked songs, lists served, and Postgres row counts.
  - **Data and pipeline:** runs, run and stage times, Ray Data throughput, and PSI per feature (latest batch and over time).
  - **Laptop and app load:** A4's CPU-vs-traffic and memory panels, plus the monitoring stack's own cost.
- **Generated, not hand-edited:**
  - `monitoring/grafana/build_dashboards.py` builds all 56 panels with the same helpers, so A4's 7 commandments hold everywhere. Okabe-Ito colours; series in a panel differ by line style too; named axes that say "linear"; a unit on every panel; a legend whenever there's more than one series; a title and a one-paragraph description on every panel.
  - `tests/test_monitoring.py` checks all of this. It also checks that the JSON matches the builder, that every query names a metric something exports (a typo can't leave an empty panel), and that promtool parses every query.
- **Series that come and go** (model versions, pipeline triggers) are shown as labelled bars, not lines. Colours fixed by name can't follow versions that don't exist yet, and Grafana's own palette isn't colour-blind safe.
- **AI attribution, as in A4:** every query starts with a `# AI-generated PromQL` comment line saying what it computes, and so does every rule in the rule files.
- **Viewing:**
  - Grafana lets anyone on the laptop view (anonymous Viewer) and allows embedding. Editing needs the admin login.
  - The console's Monitoring tab embeds six panels as iframes (`ui.GRAFANA_PANELS`; a test checks they exist), so an operator sees alerts and live charts without leaving the console. This is the rubric's "orchestrating the visualisation from multiple tools".
  - Ports are on localhost only, as for every other service.
- **Considered:** building the dashboards by hand in the Grafana UI and exporting them, as in A4. With 56 panels, keeping the colours, styles and descriptions consistent by hand would be slow, and nothing would catch a mistyped metric.
