"""Tests for the recommender, Fisher weights and the held-out split (no MLflow)."""

import numpy as np
import pandas as pd
import pytest

from spatify.model.data import eval_mask
from spatify.model.recommender import (
    EmptyQueryError,
    Recommender,
    UnknownTrackError,
    fisher_weights,
)

COLUMNS = ["energy", "danceability", "mode"]


def catalogue() -> pd.DataFrame:
    """Six tracks: two loud rock songs (one released twice), two dance tracks, one between."""
    rows = [
        # track_id, track_name, artists, genres, energy, danceability, mode
        ("a", "Song A", "X", "rock", 0.90, 0.20, 1),
        ("a2", "Song A", "X", "rock", 0.90, 0.20, 1),  # another release of Song A
        ("b", "Song B", "Y", "rock", 0.85, 0.25, 1),
        ("c", "Song C", "Z", "pop", 0.10, 0.90, 0),
        ("d", "Song D", "W", "pop;dance", 0.15, 0.85, 0),
        ("e", "Song E", "V", "jazz", 0.50, 0.50, 1),
    ]
    df = pd.DataFrame(
        rows, columns=["track_id", "track_name", "artists", "genres", *COLUMNS]
    ).assign(album_name="album", popularity=50)
    return df


def recommender(weights=None) -> Recommender:
    cat = catalogue()
    raw = cat[COLUMNS].to_numpy(dtype=float)
    weights = np.ones(len(COLUMNS)) if weights is None else weights
    return Recommender(cat, COLUMNS, raw.mean(axis=0), raw.std(axis=0), weights)


def test_seed_and_other_releases_of_its_song_are_not_recommended():
    recs = recommender().recommend(pd.DataFrame({"track_id": ["a"]}), k=10)
    assert "a" not in set(recs["track_id"])
    assert "a2" not in set(recs["track_id"])  # same artists and title as the seed
    assert recs["track_id"].iloc[0] == "b"  # the other loud rock song is closest


def test_results_are_ranked_by_similarity():
    recs = recommender().recommend(pd.DataFrame({"track_id": ["c"]}), k=4)
    assert recs["rank"].tolist() == [1, 2, 3, 4]
    assert recs["score"].is_monotonic_decreasing
    assert recs["track_id"].iloc[0] == "d"


def test_a_song_released_twice_is_recommended_once():
    recs = recommender().recommend(pd.DataFrame({"track_id": ["b"]}), k=5)
    assert recs["track_id"].isin(["a", "a2"]).sum() == 1


def test_sliders_alone_find_matching_tracks():
    # mode is left unset. Filled with its median (1, major) it would pull the major-key
    # jazz track e above the minor-key dance track d (the reason for decision D19).
    recs = recommender().recommend(pd.DataFrame({"energy": [0.1], "danceability": [0.9]}), k=2)
    assert set(recs["track_id"]) == {"c", "d"}


def test_sliders_override_the_seed_track():
    query = pd.DataFrame({"track_id": ["a"], "energy": [0.1], "danceability": [0.9]})
    recs = recommender().recommend(query, k=1)
    assert recs["track_id"].iloc[0] in {"c", "d"}  # not b, the seed's own neighbour


def test_unset_features_sit_at_the_mean():
    rec = recommender()
    raw, seeds = rec.query_features(pd.DataFrame({"energy": [0.3]}))
    assert raw[0, 0] == 0.3
    assert np.allclose(rec.transform(raw)[0, 1:], 0.0)  # z = 0: no pull either way
    assert seeds[0] == -1


def test_unknown_seed_track_is_an_error():
    with pytest.raises(UnknownTrackError, match="nope"):
        recommender().recommend(pd.DataFrame({"track_id": ["nope"]}))


def test_a_query_with_nothing_set_is_an_error():
    with pytest.raises(EmptyQueryError):
        recommender().recommend(pd.DataFrame({"track_id": [None], "energy": [np.nan]}))
    with pytest.raises(EmptyQueryError):  # tempo isn't one of this model's features
        recommender().recommend(pd.DataFrame({"tempo": [120.0]}))


def test_k_is_capped_by_what_can_be_recommended():
    recs = recommender().recommend(pd.DataFrame({"track_id": ["a"]}), k=50)
    assert len(recs) == 4  # 6 tracks minus the seed and its other release


def test_several_queries_are_numbered():
    recs = recommender().recommend(pd.DataFrame({"track_id": ["a", "c"]}), k=2)
    assert recs["query"].tolist() == [0, 0, 1, 1]


def test_saved_settings_rebuild_the_same_model():
    rec = recommender(weights=np.array([2.0, 0.5, 1.0]))
    rebuilt = Recommender.from_info(catalogue(), rec.info())
    query = pd.DataFrame({"track_id": ["e"]})
    pd.testing.assert_frame_equal(rec.recommend(query, 3), rebuilt.recommend(query, 3))


def test_fisher_weights_favour_features_that_separate_genres():
    rng = np.random.default_rng(0)
    n = 400
    genres = pd.Series(["g1"] * (n // 2) + ["g2"] * (n // 2))
    separating = np.where(genres == "g1", -1.0, 1.0) + rng.normal(0, 0.3, n)
    noise = rng.normal(0, 1, n)
    weights = fisher_weights(np.column_stack([separating, noise]), genres)
    assert weights[0] > 5 * weights[1]
    assert np.isclose(np.mean(weights**2), 1.0)


def test_eval_split_is_stable_as_the_catalogue_grows():
    ids = pd.Series([f"t{i}" for i in range(10_000)])
    mask = eval_mask(ids, 0.1, seed=42)
    assert 0.08 < mask.mean() < 0.12
    grown = eval_mask(pd.concat([ids, pd.Series(["new1", "new2"])]), 0.1, seed=42)
    assert (grown[:10_000] == mask).all()  # existing tracks keep their side
    assert not (eval_mask(ids, 0.1, seed=7) == mask).all()  # the seed changes the split
