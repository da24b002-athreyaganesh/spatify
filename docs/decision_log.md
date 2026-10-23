# Decision log

One entry per design decision: what was chosen, why, and what else was considered. These feed the HLD and the viva.

## D1. The repository lives in the course folder (`Project/spatify`) on the NTFS drive
- **Why:** it keeps the code next to the course materials, on the drive with more free space (548 GB free, against 76 GB on the ext4 home partition).
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
