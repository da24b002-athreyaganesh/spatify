"""Tests for the features stage's pandas steps (no Ray)."""

import numpy as np
import pandas as pd

from spatify.pipeline.features import add_engineered, collapse_tracks


def rows(*specs) -> pd.DataFrame:
    """Tracks from (track_id, genre, popularity, key, mode) tuples."""
    return pd.DataFrame(
        [
            {
                "track_id": track_id,
                "track_genre": genre,
                "popularity": float(popularity),
                "duration_ms": 180_000.0,
                "key": float(key),
                "mode": float(mode),
                "time_signature": 4.0,
                "energy": 0.5,
            }
            for track_id, genre, popularity, key, mode in specs
        ]
    )


def test_collapse_merges_genres_and_keeps_highest_popularity():
    tracks = collapse_tracks(
        rows(("t1", "rock", 40, 0, 1), ("t1", "pop", 55, 0, 1), ("t1", "rock", 40, 0, 1))
    )
    assert len(tracks) == 1
    track = tracks.iloc[0]
    assert track["genres"] == "pop;rock"
    assert track["n_genres"] == 2
    assert track["popularity"] == 55


def test_collapse_keeps_single_genre_tracks_apart():
    tracks = collapse_tracks(rows(("t1", "rock", 40, 0, 1), ("t2", "jazz", 30, 2, 1)))
    assert sorted(tracks["track_id"]) == ["t1", "t2"]


def key_point(key: int, mode: int) -> np.ndarray:
    track = add_engineered(rows(("t", "g", 0, key, mode))).iloc[0]
    return np.array([track["key_sin"], track["key_cos"]])


def test_circle_of_fifths_puts_related_keys_close():
    c_major, g_major, f_sharp_major = key_point(0, 1), key_point(7, 1), key_point(6, 1)
    assert np.linalg.norm(c_major - g_major) < np.linalg.norm(c_major - f_sharp_major)


def test_minor_key_sits_on_its_relative_major():
    np.testing.assert_allclose(key_point(9, 0), key_point(0, 1), atol=1e-12)  # A minor = C major


def test_integer_columns_come_back_as_integers():
    out = add_engineered(rows(("t", "g", 50, 0, 1)))
    assert out["popularity"].dtype == "int64"
    assert out["duration_min"].iloc[0] == 3.0
