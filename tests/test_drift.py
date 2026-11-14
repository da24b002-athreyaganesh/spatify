"""Drift tests: PSI and KS on distributions with a known answer, and the drift rule."""

import numpy as np
import pandas as pd
import pytest

from spatify.pipeline.drift import batch_drift, level, psi, report

CFG = {
    "reference": "tracks.csv",
    "features": ["energy", "valence"],
    "bins": 10,
    "psi_moderate": 0.1,
    "psi_major": 0.2,
    "min_moderate": 2,
}
rng = np.random.default_rng(0)


def test_psi_is_near_zero_for_the_same_distribution():
    assert psi(rng.normal(0, 1, 50_000), rng.normal(0, 1, 2_000), 10) < 0.02


def test_psi_grows_with_the_shift():
    ref = rng.normal(0, 1, 50_000)
    small, large = psi(ref, rng.normal(0.25, 1, 2_000), 10), psi(ref, rng.normal(1.0, 1, 2_000), 10)
    assert 0.02 < small < 0.2 < large


def test_psi_handles_a_feature_that_is_often_exactly_zero():
    ref = np.concatenate([np.zeros(40_000), rng.uniform(0, 1, 10_000)])  # like instrumentalness
    value = psi(ref, np.concatenate([np.zeros(800), rng.uniform(0, 1, 200)]), 10)
    assert np.isfinite(value) and value < 0.05


def test_levels_follow_the_rule_of_thumb():
    assert [level(v, CFG) for v in (0.05, 0.1, 0.19, 0.2)] == [
        "stable",
        "moderate",
        "moderate",
        "major",
    ]


def frame(energy, valence, source):
    return pd.DataFrame({"energy": energy, "valence": valence, "source": source})


def test_one_major_feature_means_drift():
    ref = frame(rng.uniform(0, 1, 20_000), rng.uniform(0, 1, 20_000), "tracks.csv")
    batch = frame(rng.uniform(0.4, 1, 2_000), rng.uniform(0, 1, 2_000), "b.csv")
    result = batch_drift(ref, batch, CFG)
    assert result["drifted"] and result["features"]["energy"]["level"] == "major"
    assert result["features"]["valence"]["level"] == "stable"
    assert result["features"]["energy"]["ks_pvalue"] < 1e-6  # KS agrees, but doesn't decide


def test_a_single_moderate_feature_is_not_drift_but_two_are():
    ref = frame(rng.uniform(0, 1, 20_000), rng.uniform(0, 1, 20_000), "tracks.csv")
    nudged = rng.uniform(0.08, 1, 2_000)  # PSI between 0.1 and 0.2
    one = batch_drift(ref, frame(nudged, rng.uniform(0, 1, 2_000), "b.csv"), CFG)
    assert one["features"]["energy"]["level"] == "moderate" and not one["drifted"]
    two = batch_drift(ref, frame(nudged, rng.uniform(0.08, 1, 2_000), "b.csv"), CFG)
    assert two["drifted"]


def test_report_checks_every_batch_against_the_reference():
    tracks = pd.concat(
        [
            frame(rng.uniform(0, 1, 5_000), rng.uniform(0, 1, 5_000), "tracks.csv"),
            frame(rng.uniform(0, 1, 500), rng.uniform(0, 1, 500), "normal.csv"),
            frame(rng.uniform(0.5, 1, 500), rng.uniform(0, 1, 500), "shifted.csv"),
        ]
    )
    result = report(tracks, CFG)
    assert result["reference_rows"] == 5_000
    assert list(result["batches"]) == ["normal.csv", "shifted.csv"]
    assert result["drifted_batches"] == ["shifted.csv"]


def test_report_without_reference_rows_stops_the_stage():
    with pytest.raises(SystemExit):
        report(frame([0.5], [0.5], "other.csv"), CFG)
