"""Why a track was recommended: the features that contribute most to its score.

The recommender ranks by cosine similarity: the sum, over features, of the query's value
times the track's value, both as weighted z-scores scaled to length 1 (D16). So each
feature's product is its exact share of the score. The labels name the features with the
largest positive shares: those where the query and the track are both well above, or
both well below, the average track ("both energetic", "both acoustic").

Kept apart from recommender.py, so the served model and its DVC stage don't change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spatify.model.recommender import Recommender

# Per feature: the word when both are above the average track, and when both are below.
WORDS = {
    "danceability": ("danceable", "less danceable"),
    "energy": ("energetic", "calm"),
    "loudness": ("loud", "quiet"),
    "speechiness": ("speech-heavy", "little speech"),
    "acousticness": ("acoustic", "electric"),
    "instrumentalness": ("instrumental", "with vocals"),
    "liveness": ("live-sounding", "studio-sounding"),
    "valence": ("upbeat", "melancholic"),
    "tempo": ("fast", "slow"),
    "mode": ("major key", "minor key"),
    "key_sin": ("related key", "related key"),
    "key_cos": ("related key", "related key"),
}


def explain(
    recommender: Recommender,
    query: pd.DataFrame,
    track_ids: list[str],
    top: int = 3,
    min_share: float = 0.05,
) -> list[list[str]]:
    """Up to `top` labels per track, strongest first; [] for a track not in the catalogue.

    `query` is one row, as the model receives it (track_id and/or sliders). A feature
    needs at least `min_share` of the score (a cosine runs from -1 to 1) to be named.
    Raises QueryError for an unknown seed or an empty query, as recommend() does.
    """
    raw, _ = recommender.query_features(query.iloc[:1])
    q = recommender.transform(raw)[0]
    norm = np.linalg.norm(q)
    if norm == 0:
        return [[] for _ in track_ids]
    q = q / norm
    rows = recommender.positions.get_indexer([str(t) for t in track_ids])
    labels = []
    for row in rows:
        if row < 0:
            labels.append([])
            continue
        shares = recommender.matrix[row].astype(np.float64) * q
        words: list[str] = []
        for j in np.argsort(-shares, kind="stable"):
            if shares[j] < min_share or len(words) == top:
                break
            high, low = WORDS.get(recommender.columns[j], (recommender.columns[j],) * 2)
            word = high if q[j] > 0 else low  # a positive share means both share the sign
            if word not in words:
                words.append(word)
        labels.append(words)
    return labels
