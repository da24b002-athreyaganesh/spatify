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
