"""Tests for the "why" labels: the features with the largest shares of the cosine score."""

import numpy as np
import pandas as pd
import pytest
from test_recommender import recommender

from spatify.model.explain import explain
from spatify.model.recommender import UnknownTrackError


def test_labels_name_traits_the_seed_and_track_share():
    # a and b are both energetic, not very danceable and in a major key (vs the catalogue).
    labels = explain(recommender(), pd.DataFrame({"track_id": ["a"]}), ["b"])
    assert set(labels[0]) == {"energetic", "less danceable", "major key"}


def test_labels_are_ordered_by_share_and_capped():
    # Weights 2/1/1 give shares energy 0.72, danceability 0.17, mode 0.11 of the score.
    rec = recommender(weights=np.array([2.0, 1.0, 1.0]))
    labels = explain(rec, pd.DataFrame({"track_id": ["a"]}), ["b"], top=2)
    assert labels == [["energetic", "less danceable"]]


def test_features_with_a_tiny_share_are_not_named():
    # Weights 3/0.5/0.5: danceability and mode drop below 5% of the score (0.025, 0.016).
    rec = recommender(weights=np.array([3.0, 0.5, 0.5]))
    assert explain(rec, pd.DataFrame({"track_id": ["a"]}), ["b"]) == [["energetic"]]


def test_slider_only_queries_explain_only_the_sliders_set():
    labels = explain(recommender(), pd.DataFrame({"energy": [0.1]}), ["c"])
    assert labels == [["calm"]]  # the other features sit at the mean and contribute nothing


def test_tracks_outside_the_catalogue_get_no_labels():
    assert explain(recommender(), pd.DataFrame({"track_id": ["a"]}), ["zzz"]) == [[]]


def test_an_unknown_seed_is_an_error_as_in_recommend():
    with pytest.raises(UnknownTrackError):
        explain(recommender(), pd.DataFrame({"track_id": ["nope"]}), ["b"])
