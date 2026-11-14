"""Tests for the validate stage's row checks and report (pandas only, no Ray)."""

import pandas as pd
import pytest

from spatify.pipeline.common import load_params
from spatify.pipeline.validate import check_rows, report

RANGES = load_params("validate")["ranges"]

GOOD = {
    "track_id": "t1",
    "artists": "A",
    "album_name": "B",
    "track_name": "C",
    "popularity": "50",
    "duration_ms": "200000",
    "explicit": "False",
    "danceability": "0.5",
    "energy": "0.5",
    "key": "5",
    "loudness": "-8.0",
    "mode": "1",
    "speechiness": "0.05",
    "acousticness": "0.2",
    "instrumentalness": "0.0",
    "liveness": "0.1",
    "valence": "0.4",
    "tempo": "120.0",
    "time_signature": "4",
    "track_genre": "pop",
    "source": "tracks.csv",
}


def checked(**changes) -> pd.Series:
    """Run the checks on one row: GOOD with some values changed."""
    return check_rows(pd.DataFrame([{**GOOD, **changes}]), RANGES).iloc[0]


def test_good_row_passes_and_is_typed():
    row = checked()
    assert row["reject_reason"] == ""
    assert row["warning"] == ""
    assert row["tempo"] == 120.0
    assert row["explicit"] is False or row["explicit"] == False  # noqa: E712 (pandas boolean)


@pytest.mark.parametrize(
    ("changes", "reason", "value"),
    [
        ({"track_id": None}, "null:track_id", ""),
        ({"tempo": None}, "null:tempo", ""),
        ({"tempo": "fast"}, "type:tempo", "fast"),
        ({"explicit": "maybe"}, "type:explicit", "maybe"),
        ({"danceability": "1.5"}, "range:danceability", "1.5"),
        ({"duration_ms": "0"}, "range:duration_ms", "0"),
    ],
)
def test_bad_rows_are_rejected_with_a_reason(changes, reason, value):
    row = checked(**changes)
    assert row["reject_reason"] == reason
    assert row["reject_value"] == value


def test_first_problem_wins():
    assert checked(track_id=None, danceability="1.5")["reject_reason"] == "null:track_id"


@pytest.mark.parametrize(
    ("changes", "warning"),
    [({"artists": None}, "missing:artists"), ({"tempo": "0", "time_signature": "0"}, "zero:tempo")],
)
def test_soft_problems_warn_but_keep_the_row(changes, warning):
    row = checked(**changes)
    assert row["reject_reason"] == ""
    assert row["warning"] == warning


def test_report_fails_only_the_bad_source():
    counts = pd.DataFrame(
        [
            ("tracks.csv", "", "", 990),
            ("tracks.csv", "range:energy", "", 10),  # 1% rejected
            ("bad.csv", "", "", 80),
            ("bad.csv", "null:tempo", "", 20),  # 20% rejected
        ],
        columns=["source", "reject_reason", "warning", "rows"],
    )
    summary = report(counts, max_reject_rate=0.05)
    assert summary["failed_sources"] == ["bad.csv"]
    assert summary["rejected"] == 30
    assert summary["reject_rate_by_source"] == {"bad.csv": 0.2, "tracks.csv": 0.01}
