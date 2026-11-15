"""Feedback drift: has the like rate fallen? (D37)

The like rate of the latest `recent` likes and skips, against the `baseline` before them.
A drop of at least `threshold` (absolute) while the audio features stay stable is the
concept-drift signal from L1: listeners react differently to the same kind of songs, or
a new model serves them worse. It is shown and alerted on; it doesn't retrain by itself,
so a bad deployment is rolled back by an operator rather than retrained over (D37).
"""

from __future__ import annotations


def feedback_drift(stats: dict, recent: int, baseline: int, threshold: float) -> dict:
    """The report from like_rates(): rates, the drop, and whether it counts as drift."""
    recent_rate, baseline_rate = stats.get("recent_rate"), stats.get("baseline_rate")
    enough = (stats.get("recent_events") or 0) >= recent and (
        stats.get("baseline_events") or 0
    ) >= baseline // 2
    drop = (
        round(float(baseline_rate) - float(recent_rate), 4)
        if recent_rate is not None and baseline_rate is not None
        else None
    )
    return {
        "recent_events": int(stats.get("recent_events") or 0),
        "recent_like_rate": None if recent_rate is None else round(float(recent_rate), 4),
        "baseline_events": int(stats.get("baseline_events") or 0),
        "baseline_like_rate": None if baseline_rate is None else round(float(baseline_rate), 4),
        "drop": drop,
        "threshold": threshold,
        "enough_feedback": enough,
        "drifted": bool(enough and drop is not None and drop >= threshold),
    }
