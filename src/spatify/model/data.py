"""Loading the feature table and splitting off held-out evaluation tracks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def load_tracks(path: Path = Path("data/features/tracks")) -> pd.DataFrame:
    """The features stage's track table (a folder of Parquet files), in track_id order."""
    tracks = pd.read_parquet(path)
    return tracks.sort_values("track_id", kind="stable").reset_index(drop=True)


def eval_mask(track_ids: pd.Series, fraction: float, seed: int) -> np.ndarray:
    """True for held-out evaluation tracks.

    A track is held out when a hash of (seed, track_id) falls in the first `fraction` of
    the hash range. Unlike a random sample, this keeps every track on the same side of the
    split when new batches grow the catalogue, so an evaluation track never becomes a
    training track later.
    """

    def unit(track_id: str) -> float:
        digest = hashlib.md5(f"{seed}:{track_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    return track_ids.astype(str).map(unit).to_numpy() < fraction


def eval_queries(tracks: pd.DataFrame, split: dict, n: int, seed: int) -> list[str]:
    """A seeded sample of n held-out track_ids, used as seed tracks for evaluation."""
    is_eval = eval_mask(tracks["track_id"], split["eval_fraction"], split["seed"])
    held_out = tracks.loc[is_eval, "track_id"]
    held_out = held_out.sort_values().to_numpy()
    rng = np.random.default_rng(seed)
    return rng.choice(held_out, size=min(n, len(held_out)), replace=False).tolist()


def genre_sets(tracks: pd.DataFrame) -> dict[str, set[str]]:
    """track_id -> the set of its genres."""
    return dict(zip(tracks["track_id"], tracks["genres"].str.split(";").map(set), strict=True))
