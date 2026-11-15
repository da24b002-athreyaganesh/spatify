"""Simulated listeners: the source of most of Spatify's feedback (D38).

Real listeners are few, so likes and skips mostly come from these simulated ones, and the
report says so. Each listener has hidden preferences: how much they care about each of a
few audio features, and how close a track must be to what they want before they like it.
The system never sees TASTE; it has to learn it from likes and skips (D37).

A listener starting from a seed song wants songs like it on the features they care about;
one using the sliders wants those values. They like a track with a probability that falls
as it moves away from what they want, so the feedback is noisy, as real feedback is.

scripts/simulate_users.py sends these listeners through the real gateway; `offline_lists`
runs them against a model directly, for tests and scripts/feedback_experiment.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from spatify.model.feedback import EVENT_COLUMNS

# What simulated listeners care about, and how much (hidden from the system).
TASTE = {"energy": 3.0, "valence": 3.0, "danceability": 2.0, "acousticness": 2.0, "tempo": 1.0}
# Divisors that put raw values on a 0-1 scale (the 0-1 features need none).
SCALE = {"tempo": 200.0}
SLOPE = 25.0  # how sharply the like probability falls as a track moves away


@dataclass
class Listener:
    importance: dict[str, float]  # per feature; a feature not listed doesn't matter
    tolerance: float  # how far a track may be (0-1 scale) for an even chance of a like

    @classmethod
    def new(cls, rng: np.random.Generator, features: list[str] | None = None) -> Listener:
        """A listener with personal variations on TASTE, over `features` (default: all)."""
        chosen = features or list(TASTE)
        importance = {f: TASTE[f] * float(rng.lognormal(0, 0.3)) for f in chosen}
        # Calibrated so about 4 in 10 cold-start recommendations are liked (D38).
        return cls(importance, float(rng.uniform(0.04, 0.09)))

    def distance(self, target: dict[str, float], track: dict[str, float]) -> float:
        """Importance-weighted mean gap between a track and what the listener wants."""
        total = sum(self.importance.values())
        gaps = sum(
            weight * abs(track[f] - target[f]) / SCALE.get(f, 1.0)
            for f, weight in self.importance.items()
        )
        return gaps / total

    def like_probability(self, target: dict[str, float], track: dict[str, float]) -> float:
        return 1.0 / (1.0 + math.exp(-SLOPE * (self.tolerance - self.distance(target, track))))

    def rate(self, target: dict, track: dict, rng: np.random.Generator) -> str:
        return "like" if rng.random() < self.like_probability(target, track) else "skip"


def slider_targets(rng: np.random.Generator) -> dict[str, float]:
    """Two or three slider values a listener might ask for, in the gateway's units."""
    chosen = rng.choice(list(TASTE), size=int(rng.integers(2, 4)), replace=False)
    targets = {}
    for feature in chosen:
        if feature == "tempo":
            targets[feature] = float(rng.choice([80.0, 100.0, 120.0, 140.0]))
        else:
            targets[feature] = float(np.round(rng.uniform(0.1, 0.9), 2))
    return targets


def new_query(rng: np.random.Generator, track_ids: np.ndarray, slider_share: float) -> dict:
    """What a new listener asks for: a seed song, or (with probability `slider_share`) two
    or three sliders. The body of a /recommend request, without k."""
    if rng.random() < slider_share:
        return {"sliders": slider_targets(rng)}
    return {"track_id": str(rng.choice(track_ids))}


def listener_for(
    query: dict, sounds: pd.DataFrame, rng: np.random.Generator
) -> tuple[Listener, dict[str, float]]:
    """A listener for `query`, and what they want: the seed song's values on the features
    they care about, or the slider values they set (and then they care about only those).

    `sounds` holds the raw TASTE features, indexed by track_id: how each song sounds.
    """
    if "track_id" in query:
        return Listener.new(rng), sounds.loc[query["track_id"], list(TASTE)].to_dict()
    target = dict(query["sliders"])
    return Listener.new(rng, list(target)), target


def rate_list(
    listener: Listener,
    target: dict[str, float],
    track_ids: list[str],
    sounds: pd.DataFrame,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """(track_id, "like" or "skip") for each track the listener can hear, in list order.
    A track missing from `sounds` (newer than the simulator's catalogue) isn't rated."""
    return [
        (t, listener.rate(target, sounds.loc[t].to_dict(), rng))
        for t in track_ids
        if t in sounds.index
    ]


def offline_lists(
    model,
    sounds: pd.DataFrame,
    lists: int,
    rng: np.random.Generator,
    slider_share: float = 0.3,
    k: int = 10,
) -> pd.DataFrame:
    """Likes and skips without the stack, in the feedback snapshot's columns.

    `model` is a Recommender. Each list is its top k-1 plus one random track in the last
    slot, as the gateway serves a list of 10 (D37); `explored` marks that track.
    """
    ids = sounds.index.to_numpy()
    start, events = pd.Timestamp("2026-11-15", tz="UTC"), []
    for i in range(lists):
        query = new_query(rng, ids, slider_share)
        row = pd.DataFrame([{"track_id": query.get("track_id"), **query.get("sliders", {})}])
        shown = model.recommend(row, k)["track_id"].astype(str).tolist()
        pick = str(rng.choice(ids))
        while pick in shown or pick == query.get("track_id"):
            pick = str(rng.choice(ids))
        shown[-1] = pick
        listener, target = listener_for(query, sounds, rng)
        for track_id, action in rate_list(listener, target, shown, sounds, rng):
            events.append(
                {
                    "request_id": f"list-{i}",
                    "created_at": start + pd.Timedelta(seconds=i),
                    "seed_track_id": query.get("track_id"),
                    "sliders": json.dumps(query.get("sliders", {})),
                    "track_id": track_id,
                    "rank": shown.index(track_id) + 1,
                    "action": action,
                    "explored": track_id == pick,
                    "model_version": None,
                }
            )
    return pd.DataFrame(events, columns=EVENT_COLUMNS)
