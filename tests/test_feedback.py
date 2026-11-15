"""Learning from likes and skips (D37), on a synthetic catalogue and simulated listeners.

The listeners care about TASTE's five features and not about liveness or speechiness;
the feedback model has to find that out from their likes and skips alone.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatify.model.feedback import (
    EVENT_COLUMNS,
    fit_weights,
    load_events,
    random_ranking_metrics,
    ranking_metrics,
    split_by_time,
)
from spatify.model.recommender import DETAILS, Recommender
from spatify.simulation import TASTE, offline_lists

COLUMNS = ["danceability", "energy", "acousticness", "valence", "tempo", "liveness", "speechiness"]
IGNORED = ["liveness", "speechiness"]  # listeners don't care about these


def catalogue(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {c: rng.uniform(0, 1, n) for c in COLUMNS}
    data["tempo"] = rng.uniform(60, 180, n)
    tracks = pd.DataFrame(data)
    tracks.insert(0, "track_id", [f"t{i}" for i in range(n)])
    tracks["track_name"] = [f"Song {i}" for i in range(n)]
    tracks["artists"] = [f"Artist {i}" for i in range(n)]
    tracks["album_name"] = "LP"
    tracks["genres"] = "pop"
    tracks["popularity"] = 50
    return tracks[DETAILS + COLUMNS]


def recommender(tracks: pd.DataFrame, weights=None) -> Recommender:
    raw = tracks[COLUMNS].to_numpy(dtype=np.float64)
    weights = np.ones(len(COLUMNS)) if weights is None else weights
    return Recommender(tracks, COLUMNS, raw.mean(axis=0), raw.std(axis=0), weights)


@pytest.fixture(scope="module")
def loop():
    tracks = catalogue()
    cold = recommender(tracks)
    sounds = tracks.set_index("track_id")[list(TASTE)]
    events = offline_lists(cold, sounds, 300, np.random.default_rng(0))
    train, test = split_by_time(events, 0.2)
    weights, report = fit_weights(cold, train, min_events=1000)
    return tracks, cold, train, test, weights, report


def test_the_split_keeps_lists_whole_and_holds_out_the_latest():
    events = pd.DataFrame(
        {
            "request_id": ["a", "a", "b", "c", "c", "d", "e"],
            "created_at": pd.to_datetime([1, 1, 2, 3, 3, 4, 5], unit="s", utc=True),
        }
    )
    train, test = split_by_time(events, 0.4)
    assert set(train["request_id"]) == {"a", "b", "c"} and set(test["request_id"]) == {"d", "e"}
    assert split_by_time(events, 0.0)[1].empty


def test_offline_lists_look_like_the_gateways_snapshot(loop):
    _, _, train, test, _, _ = loop
    events = pd.concat([train, test])
    assert list(events.columns) == EVENT_COLUMNS
    per_list = events.groupby("request_id")
    assert (per_list.size() == 10).all() and (per_list["explored"].sum() == 1).all()
    assert (events.loc[events["explored"], "rank"] == 10).all()
    # Both likes and skips. (This sparse catalogue gives ~18%; the real one ~40%, which
    # scripts/feedback_experiment.py measures.)
    assert 0.05 < (events.loc[~events["explored"], "action"] == "like").mean() < 0.6


def test_a_missing_snapshot_is_an_empty_table(tmp_path):
    events = load_events(tmp_path / "nothing.parquet")
    assert events.empty and list(events.columns) == EVENT_COLUMNS


def test_the_weights_find_what_listeners_care_about(loop):
    _, _, _, _, weights, report = loop
    assert weights is not None and report["events"] >= 1000
    learned = dict(zip(COLUMNS, weights, strict=True))
    for feature in ("energy", "valence", "danceability", "acousticness"):
        assert learned[feature] > 2 * max(learned[f] for f in IGNORED), learned
    assert np.mean(weights**2) == pytest.approx(1.0)  # same overall scale as uniform weights
    assert set(report["coefficients"]) == set(COLUMNS)


def test_the_feedback_model_ranks_held_out_likes_higher_than_cold_start(loop):
    tracks, cold, _, test, weights, _ = loop
    before = ranking_metrics(cold, test, 3)
    after = ranking_metrics(recommender(tracks, weights), test, 3)
    floor = random_ranking_metrics(test, 3, seed=0)
    assert after["lists"] == before["lists"] >= 30
    assert after["ndcg"] > before["ndcg"] > floor["ndcg"]


def test_too_little_or_one_sided_feedback_gives_no_weights(loop):
    _, cold, train, _, _, _ = loop
    weights, report = fit_weights(cold, train, min_events=len(train) + 1)
    assert weights is None and report["events"] == len(train)
    only_likes = train.assign(action="like")
    assert fit_weights(cold, only_likes, min_events=10)[0] is None


def test_ranking_metrics_score_a_perfect_and_a_backwards_order():
    tracks = catalogue(50)
    model = recommender(tracks)
    seed = "t0"
    order = model.recommend(pd.DataFrame([{"track_id": seed}]), 4)["track_id"].tolist()
    rated = pd.DataFrame(
        {
            "request_id": "r",
            "created_at": pd.Timestamp("2026-11-15", tz="UTC"),
            "seed_track_id": seed,
            "sliders": "{}",
            "track_id": order,
            "action": ["like", "skip", "skip", "skip"],
        }
    )
    assert ranking_metrics(model, rated, 3) == {"hit_rate": 1.0, "ndcg": 1.0, "lists": 1}
    backwards = rated.assign(action=["skip", "skip", "skip", "like"])
    worst = ranking_metrics(model, backwards, 3)
    assert worst["hit_rate"] == 0.0 and worst["ndcg"] == 0.0
    assert ranking_metrics(model, rated.assign(action="skip"), 3)["lists"] == 0  # no like


def test_the_forced_bad_model_ranks_no_better_than_chance(loop):
    """scripts/force_bad_model.py's scrambled copy (D39) loses what the model knew."""
    tracks, cold, _, test, _, _ = loop
    path = Path(__file__).parents[1] / "scripts/force_bad_model.py"
    spec = importlib.util.spec_from_file_location("force_bad_model", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    bad = script.scrambled(cold, seed=0)
    assert list(bad.catalogue["track_id"]) == list(cold.catalogue["track_id"])
    assert not np.allclose(bad.raw, cold.raw)
    assert ranking_metrics(bad, test, 3)["ndcg"] < ranking_metrics(cold, test, 3)["ndcg"]
