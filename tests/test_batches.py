"""Tests for the batch generator, on a small synthetic catalogue."""

import numpy as np
import pandas as pd
import pytest

from spatify.batches import build

CFG = {
    "seed": 7,
    "held_out_genres": 2,
    "base_fraction": 0.8,
    "batch_size": 20,
    "n_normal": 2,
    "n_shifted": 1,
    "n_drifted": 1,
    "corrupt_fraction": 0.1,
    "drift": {"energy": 0.2},
}


@pytest.fixture
def catalogue() -> pd.DataFrame:
    """1,000 tracks spread evenly over 10 genres."""
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame(
        {
            "track_id": [f"t{i}" for i in range(n)],
            "track_genre": [f"g{i % 10}" for i in range(n)],
            "danceability": rng.uniform(0, 1, n),
            "energy": rng.uniform(0, 1, n),
            "valence": rng.uniform(0, 1, n),
            "tempo": rng.uniform(60, 200, n),
        }
    )


def test_same_seed_gives_same_output(catalogue):
    base_a, batches_a, _ = build(catalogue, CFG)
    base_b, batches_b, _ = build(catalogue, CFG)
    pd.testing.assert_frame_equal(base_a, base_b)
    for name in batches_a:
        pd.testing.assert_frame_equal(batches_a[name], batches_b[name])


def test_held_out_genres_only_in_shifted_batches(catalogue):
    base, batches, manifest = build(catalogue, CFG)
    held_out = set(manifest["held_out_genres"])
    assert not held_out & set(base["track_genre"])
    assert set(batches["shifted_01"]["track_genre"]) <= held_out
    assert not held_out & set(batches["normal_01"]["track_genre"])


def test_no_track_in_both_base_and_a_batch(catalogue):
    base, batches, _ = build(catalogue, CFG)
    batch_ids = set(pd.concat(list(batches.values()))["track_id"].dropna())
    assert not batch_ids & set(base["track_id"])


def test_drifted_batch_has_higher_energy(catalogue):
    base, batches, _ = build(catalogue, CFG)
    assert batches["drifted_01"]["energy"].mean() > base["energy"].mean() + 0.1


def test_planted_rows_match_manifest(catalogue):
    _, batches, manifest = build(catalogue, CFG)
    for entry in manifest["batches"]:
        batch = batches[entry["name"]]
        unit = batch[["danceability", "energy", "valence"]]
        out_of_range = ((unit < 0) | (unit > 1)).any(axis=1) | (batch["tempo"] < 0)
        assert batch.isna().any(axis=1).sum() == entry["planted"]["null"]
        assert out_of_range.sum() == entry["planted"]["out_of_range"]


def test_too_few_rows_raises(catalogue):
    with pytest.raises(ValueError, match="not enough"):
        build(catalogue, {**CFG, "batch_size": 500})
