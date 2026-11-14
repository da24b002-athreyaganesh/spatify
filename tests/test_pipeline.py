"""End-to-end test of the three Ray stages on a small generated catalogue."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import ray

from spatify.batches import build as build_batches
from spatify.pipeline import features, ingest, validate

PARAMS = Path("params.yaml").resolve()  # the real thresholds, read from the project root


@pytest.fixture(scope="module")
def ray_local():
    # "local": a fresh instance even if /tmp/ray records another (possibly dead) cluster.
    ray.init(
        address="local",
        num_cpus=2,
        object_store_memory=256 * 1024**2,
        include_dashboard=False,
        log_to_driver=False,
    )
    yield
    ray.shutdown()


def make_workspace(path: Path, corrupt_fraction: float) -> None:
    """A tiny project: base set and one batch in incoming/, built by the batch generator."""
    n = 600
    raw = pd.DataFrame(
        {
            "track_id": [f"t{i % 500}" for i in range(n)],  # 100 tracks appear twice
            "artists": "A",
            "album_name": "B",
            "track_name": [f"song {i}" for i in range(n)],
            "popularity": 50,
            "duration_ms": 200_000,
            "explicit": False,
            "danceability": 0.5,
            "energy": [i / n for i in range(n)],
            "key": [i % 12 for i in range(n)],
            "loudness": -8.0,
            "mode": 1,
            "speechiness": 0.05,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "valence": 0.4,
            "tempo": 120.0,
            "time_signature": 4,
            "track_genre": [f"g{i % 6}" for i in range(n)],
        }
    )
    cfg = {
        "seed": 1,
        "held_out_genres": 2,
        "base_fraction": 0.8,
        "batch_size": 40,
        "n_normal": 1,
        "n_shifted": 0,
        "n_drifted": 0,
        "corrupt_fraction": corrupt_fraction,
        "drift": {},
    }
    base, batches, _ = build_batches(raw, cfg)
    (path / "data/incoming").mkdir(parents=True)
    base.to_csv(path / "data/base.csv", index=False)
    batches["normal_01"].to_csv(path / "data/incoming/normal_01.csv", index=False)
    shutil.copy(PARAMS, path / "params.yaml")


def test_stages_run_end_to_end(ray_local, tmp_path, monkeypatch):
    make_workspace(tmp_path, corrupt_fraction=0.025)  # 1 bad row in 40: under the 5% limit
    monkeypatch.chdir(tmp_path)
    ingest.main(["--base", "data/base.csv"])
    validate.main([])
    features.main([])

    checks = json.loads(Path("metrics/validate.json").read_text())
    assert checks["rejected_by_source"]["normal_01.csv"] == 1
    assert checks["failed_sources"] == []
    tracks = pd.read_parquet("data/features/tracks")
    assert not tracks["track_id"].duplicated().any()
    assert len(tracks) < checks["valid"]  # tracks in two genres were merged
    assert Path("data/features/scaler.json").exists()
    assert Path("data/features/reference.parquet").exists()


def test_a_bad_batch_stops_the_pipeline(ray_local, tmp_path, monkeypatch):
    make_workspace(tmp_path, corrupt_fraction=0.25)  # 10 bad rows in 40
    monkeypatch.chdir(tmp_path)
    ingest.main(["--base", "data/base.csv"])
    with pytest.raises(SystemExit):
        validate.main([])
    rejected = pd.read_csv("data/validated/rejected.csv")
    assert len(rejected) == 10  # the evidence is still written for the operator
