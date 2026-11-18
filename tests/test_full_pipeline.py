"""Integration test: the whole pipeline on a small sample, stage by stage in the order
`dvc repro` runs them (ingest -> validate -> features and drift_check -> train -> evaluate
-> register), into a throwaway MLflow registry.

Then the registered champion answers a query, and registering an equally good model
again doesn't promote it: the gate (D22) only lets a better model replace the champion.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import ray
import yaml

from spatify.model import evaluate, register, train
from spatify.pipeline import drift, features, ingest, validate

PARAMS = Path("params.yaml").resolve()  # the real settings, with the overrides below
GENRES = ["pop", "metal", "ambient", "hip-hop", "folk", "edm"]
CENTRES = np.random.default_rng(7).uniform(0.1, 0.9, (len(GENRES), 6))  # per genre


@pytest.fixture(scope="module")
def ray_local():
    ray.init(
        address="local",
        num_cpus=2,
        object_store_memory=256 * 1024**2,
        include_dashboard=False,
        log_to_driver=False,
    )
    yield
    ray.shutdown()


def tracks(n: int, start: int, rng: np.random.Generator) -> pd.DataFrame:
    """Songs whose sound depends on their genre, so genre match means something."""
    genre = rng.integers(0, len(GENRES), n)
    centre = CENTRES[genre]  # each genre's typical sound, shared by the base and the batch

    def feature(j: int) -> np.ndarray:
        return np.clip(centre[:, j] + rng.normal(0, 0.08, n), 0, 1).round(3)

    return pd.DataFrame(
        {
            "track_id": [f"t{start + i}" for i in range(n)],
            "artists": [f"Artist {(start + i) % 97}" for i in range(n)],
            "album_name": "Album",
            "track_name": [f"Song {start + i}" for i in range(n)],
            "popularity": rng.integers(0, 100, n),
            "duration_ms": rng.integers(120_000, 300_000, n),
            "explicit": False,
            "danceability": feature(0),
            "energy": feature(1),
            "key": rng.integers(0, 12, n),
            "loudness": (-25 + 20 * feature(2)).round(2),
            "mode": rng.integers(0, 2, n),
            "speechiness": (feature(3) / 3).round(3),
            "acousticness": feature(4),
            "instrumentalness": (feature(5) / 2).round(3),
            "liveness": rng.uniform(0.05, 0.4, n).round(3),
            "valence": feature(0)[::-1],
            "tempo": rng.uniform(70, 170, n).round(1),
            "time_signature": 4,
            "track_genre": [GENRES[g] for g in genre],
        }
    )


def make_project(path: Path) -> None:
    """A base catalogue, one normal batch in data/incoming and a trimmed params.yaml."""
    base = tracks(900, 0, np.random.default_rng(1))
    batch = tracks(400, 10_000, np.random.default_rng(2))  # big enough for a steady PSI
    (path / "data/incoming").mkdir(parents=True)
    base.to_csv(path / "data/base.csv", index=False)
    batch.to_csv(path / "data/incoming/normal_01.csv", index=False)
    params = yaml.safe_load(PARAMS.read_text())
    params["drift"]["reference"] = "base.csv"  # the base file's name in this project
    params["train"]["feature_sets"] = {"all": []}  # 2 candidates instead of 6: faster
    (path / "params.yaml").write_text(yaml.safe_dump(params))


def test_the_whole_pipeline_trains_registers_and_serves_a_champion(
    ray_local, tmp_path, monkeypatch
):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("SPATIFY_TRIGGER", "integration_test")

    ingest.main(["--base", "data/base.csv"])
    validate.main([])
    features.main([])
    drift.main([])
    train.main([])
    evaluate.main([])
    register.main([])

    checks = json.loads(Path("metrics/validate.json").read_text())
    assert checks["failed_sources"] == [] and checks["valid"] == 1300
    batches = json.loads(Path("metrics/drift.json").read_text())["batches"]
    assert batches["normal_01.csv"]["drifted"] is False  # same sound as the base set
    candidates = json.loads(Path("models/candidates.json").read_text())["candidates"]
    assert {c["variant"] for c in candidates} == {"all-uniform", "all-fisher"}
    scores = json.loads(Path("metrics/evaluate.json").read_text())
    assert scores["best"]["genre_match_at_k"] > 2 * scores["random_baseline"]

    first = json.loads(Path("metrics/register.json").read_text())
    assert first["version"] == 1 and first["promoted"] is True  # no champion yet

    model = mlflow.pyfunc.load_model("models:/spatify@champion")
    answer = model.predict(pd.DataFrame([{"track_id": "t5"}]), params={"k": 5})
    assert len(answer) == 5 and "t5" not in set(answer["track_id"])
    assert set(answer["model_run_id"]) == {
        next(c["run_id"] for c in candidates if c["variant"] == first["variant"])
    }

    register.main([])  # the same best model again: no better, so not promoted
    second = json.loads(Path("metrics/register.json").read_text())
    assert second["version"] == 2 and second["promoted"] is False
    assert second["gain"] == 0 and second["champion_version"] == 1
