# Decision log

One entry per design decision: what was chosen, why, and what else was considered. These feed the HLD and the viva.

## D1. The repository lives in the course folder (`Project/spatify`) on the NTFS drive
- **Why:** it keeps the code next to other course materials and keeps my stuff organised.
- **Costs and mitigations:** NTFS (`ntfs3`) doesn't keep Unix file modes, so Git's `core.fileMode` is off and executable bits are set in the Dockerfiles rather than taken from the checkout. uv can't hardlink from its cache on ext4, so `link-mode = "copy"` is set. Service data (Postgres, MLflow, Prometheus, Grafana) lives in named Docker volumes, which sit on ext4.
- **Considered:** `~/spatify` on ext4, which avoids these workarounds.
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
