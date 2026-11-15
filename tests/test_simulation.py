"""The simulated listeners (D38): how they ask, what they want, how they rate."""

import numpy as np
import pandas as pd
import pytest

from spatify.simulation import TASTE, Listener, listener_for, new_query, rate_list, slider_targets

SOUNDS = pd.DataFrame(
    {
        "energy": [0.8, 0.8, 0.1],
        "valence": [0.5, 0.5, 0.9],
        "danceability": [0.7, 0.7, 0.2],
        "acousticness": [0.1, 0.1, 0.9],
        "tempo": [120.0, 122.0, 70.0],
    },
    index=pd.Index(["seed", "twin", "opposite"], name="track_id"),
)


def test_the_like_probability_falls_with_distance_and_is_even_at_the_tolerance():
    listener = Listener({"energy": 1.0}, tolerance=0.1)
    target = {"energy": 0.5}
    near, edge, far = (listener.like_probability(target, {"energy": e}) for e in (0.5, 0.6, 0.9))
    assert near > edge > far
    assert edge == pytest.approx(0.5)


def test_tempo_gaps_are_scaled_like_the_unit_features():
    listener = Listener({"tempo": 1.0}, tolerance=0.1)
    assert listener.distance({"tempo": 100.0}, {"tempo": 120.0}) == pytest.approx(0.1)


def test_a_new_listener_varies_the_shared_taste_over_the_chosen_features():
    rng = np.random.default_rng(0)
    everyone = Listener.new(rng)
    assert set(everyone.importance) == set(TASTE)
    assert 0.04 <= everyone.tolerance <= 0.09
    assert set(Listener.new(rng, ["energy"]).importance) == {"energy"}


def test_queries_are_seeds_or_two_or_three_sliders_in_the_gateways_units():
    rng = np.random.default_rng(0)
    ids = np.array(["seed", "twin"])
    assert all("track_id" in new_query(rng, ids, 0.0) for _ in range(20))
    for _ in range(20):
        sliders = new_query(rng, ids, 1.0)["sliders"]
        assert 2 <= len(sliders) <= 3 and set(sliders) <= set(TASTE)
        assert all(0 <= v <= 1 for f, v in sliders.items() if f != "tempo")
    assert slider_targets(np.random.default_rng(1)) == slider_targets(np.random.default_rng(1))


def test_a_seed_listener_wants_the_seeds_sound_and_a_slider_listener_only_the_sliders():
    rng = np.random.default_rng(0)
    listener, target = listener_for({"track_id": "seed"}, SOUNDS, rng)
    assert target == SOUNDS.loc["seed"].to_dict() and set(listener.importance) == set(TASTE)
    listener, target = listener_for({"sliders": {"energy": 0.3}}, SOUNDS, rng)
    assert target == {"energy": 0.3} and set(listener.importance) == {"energy"}


def test_listeners_mostly_like_the_seeds_twin_and_skip_its_opposite():
    listener = Listener(dict(TASTE), tolerance=0.06)
    target = SOUNDS.loc["seed"].to_dict()
    twin = listener.like_probability(target, SOUNDS.loc["twin"].to_dict())
    opposite = listener.like_probability(target, SOUNDS.loc["opposite"].to_dict())
    assert 0.75 < twin < 1.0  # even a near-identical song isn't a sure like: feedback is noisy
    assert opposite < 0.01
    ratings = rate_list(listener, target, ["twin", "opposite"], SOUNDS, np.random.default_rng(0))
    assert [t for t, _ in ratings] == ["twin", "opposite"]
    assert {a for _, a in ratings} <= {"like", "skip"}


def test_tracks_the_listener_cant_hear_are_not_rated():
    listener = Listener({"energy": 1.0}, tolerance=0.1)
    rated = rate_list(listener, {"energy": 0.8}, ["new", "twin"], SOUNDS, np.random.default_rng(0))
    assert [t for t, _ in rated] == ["twin"]
