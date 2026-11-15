"""Learning from likes and skips, and judging models by them (D37).

Feedback events come from `data/feedback/feedback.parquet`, a DVC-tracked snapshot the
worker exports from Postgres before each retrain: one row per like or skip, with the
list's query (seed track and sliders) and the track it was about.

- `fit_weights`: logistic regression on the squared differences between query and track,
  feature by feature, as z-scores. A negative coefficient on feature j means listeners
  like tracks that are close to the query on j. The weight is sqrt(-coefficient), scaled
  to a mean square of 1 like the other weightings; a feature whose closeness doesn't help
  gets weight 0. (Z-scores recovered the simulated listeners' taste better than the unit
  vectors the recommender ranks with, which tie every feature to all the others.)
- `ranking_metrics`: re-rank each held-out list's rated tracks by a model's similarity,
  then hit rate@k (a like in the top k) and NDCG@k (likes count 1, skips 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from spatify.model.recommender import SLIDERS, Recommender

EVENT_COLUMNS = [
    "request_id",
    "created_at",
    "seed_track_id",
    "sliders",
    "track_id",
    "rank",
    "action",
    "explored",
    "model_version",
]


def load_events(path: Path) -> pd.DataFrame:
    """The feedback snapshot, or an empty table if there isn't one."""
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame(columns=EVENT_COLUMNS)


def split_by_time(events: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earlier lists, test on the latest `test_fraction` of them."""
    if events.empty:
        return events, events
    order = events.groupby("request_id")["created_at"].min().sort_values(kind="stable").index
    n_test = int(round(len(order) * test_fraction))
    test_ids = set(order[len(order) - n_test :]) if n_test else set()
    is_test = events["request_id"].isin(test_ids)
    return events[~is_test], events[is_test]


def _query(row: pd.Series) -> dict:
    """The query a list was made for, as the model receives it."""
    sliders = row["sliders"]
    query = dict(json.loads(sliders) if isinstance(sliders, str) else sliders or {})
    if isinstance(row["seed_track_id"], str) and row["seed_track_id"]:
        query["track_id"] = row["seed_track_id"]
    return query


def _usable(rec: Recommender, query: dict) -> bool:
    seed = query.get("track_id")
    if seed is not None:
        return seed in rec.positions
    return any(k in rec.columns for k in query if k in SLIDERS)


def query_vectors(rec: Recommender, lists: pd.DataFrame, unit: bool = True) -> dict:
    """request_id -> the query in `rec`'s weighted z-score space, as a unit vector unless
    `unit` is False (lists whose seed or sliders `rec` can't read are left out)."""
    vectors = {}
    for request_id, row in lists.groupby("request_id").first().iterrows():
        query = _query(row)
        if not _usable(rec, query):
            continue
        raw, _ = rec.query_features(pd.DataFrame([query]))
        vector = rec.transform(raw)[0]
        norm = np.linalg.norm(vector)
        if not unit:
            vectors[request_id] = vector
        elif norm > 0:
            vectors[request_id] = vector / norm
    return vectors


def fit_weights(
    base: Recommender, events: pd.DataFrame, min_events: int, c: float = 1.0
) -> tuple[np.ndarray | None, dict]:
    """Per-feature weights learned from feedback, and a report (coefficients, counts).

    `base` must have uniform weights, so the squared differences compare features on an
    equal footing. Returns None for the weights with too few usable events, only likes or
    only skips, or no feature whose closeness raises the chance of a like.
    """
    vectors = query_vectors(base, events, unit=False)
    rows = events[events["request_id"].isin(vectors)]
    positions = base.positions.get_indexer(rows["track_id"].astype(str))
    rows, positions = rows[positions >= 0], positions[positions >= 0]
    report = {"events": len(rows), "likes": int((rows["action"] == "like").sum())}
    if len(rows) < min_events or report["likes"] in (0, len(rows)):
        return None, report
    queries = np.stack([vectors[r] for r in rows["request_id"]])
    tracks = base.transform(base.raw[positions])
    x = (queries - tracks) ** 2
    y = (rows["action"] == "like").to_numpy()
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(x, y)
    coefficients = model.coef_[0]
    report["coefficients"] = dict(zip(base.columns, coefficients.round(4).tolist(), strict=True))
    strength = np.clip(-coefficients, 0.0, None)
    if not strength.any():
        return None, report
    weights = np.sqrt(strength)
    return weights / np.sqrt(np.mean(weights**2)), report


def ranking_metrics(rec: Recommender, events: pd.DataFrame, k: int) -> dict:
    """hit rate@k and NDCG@k over held-out lists with at least one like and two rated tracks."""
    vectors = query_vectors(rec, events)
    hits, ndcgs = [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for request_id, rated in events[events["request_id"].isin(vectors)].groupby("request_id"):
        positions = rec.positions.get_indexer(rated["track_id"].astype(str))
        keep = positions >= 0
        liked = (rated["action"] == "like").to_numpy()[keep]
        if keep.sum() < 2 or not liked.any():
            continue
        scores = rec.matrix[positions[keep]].astype(np.float64) @ vectors[request_id]
        top = liked[np.argsort(-scores, kind="stable")][:k]
        ideal = np.sort(liked)[::-1][:k]
        hits.append(float(top.any()))
        ndcgs.append(
            float((top * discounts[: len(top)]).sum() / (ideal * discounts[: len(ideal)]).sum())
        )
    return {
        "hit_rate": round(float(np.mean(hits)), 4) if hits else None,
        "ndcg": round(float(np.mean(ndcgs)), 4) if ndcgs else None,
        "lists": len(hits),
    }


def random_ranking_metrics(events: pd.DataFrame, k: int, seed: int, rounds: int = 20) -> dict:
    """hit rate@k and NDCG@k for random orderings of the same lists: the floor to beat."""
    rng = np.random.default_rng(seed)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    hits, ndcgs = [], []
    for _, rated in events.groupby("request_id"):
        liked = (rated["action"] == "like").to_numpy()
        if len(liked) < 2 or not liked.any():
            continue
        ideal = np.sort(liked)[::-1][:k]
        best = (ideal * discounts[: len(ideal)]).sum()
        for _ in range(rounds):
            top = rng.permutation(liked)[:k]
            hits.append(float(top.any()))
            ndcgs.append(float((top * discounts[: len(top)]).sum() / best))
    return {
        "hit_rate": round(float(np.mean(hits)), 4) if hits else None,
        "ndcg": round(float(np.mean(ndcgs)), 4) if ndcgs else None,
    }
