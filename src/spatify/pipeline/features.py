"""Features stage: one row per track, engineered features, scaling, drift reference.

- A track listed under several genres becomes one row with all of them in `genres`
  (";"-separated); otherwise the recommender could return the same track twice. Audio
  features are identical across those rows; popularity can differ, so the highest is kept.
- Engineered features: the key's position on the circle of fifths as sin/cos, so
  harmonically close keys (C and G) sit close together. A minor key uses its relative
  major (A minor -> C major), which has the same notes.
- The feature columns in params.yaml are standardised with Ray's StandardScaler into
  `z_<column>`. The fitted means and standard deviations go to scaler.json, so serving
  applies exactly the same transform (no training-serving skew).
- A seeded sample of the unscaled feature values is saved as the reference for drift checks.

Run:  python -m spatify.pipeline.features
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import ray.data as rd
from ray.data.preprocessors import StandardScaler

from spatify.pipeline.common import (
    fresh_dir,
    init_ray,
    load_params,
    setup_logging,
    throughput,
    write_metrics,
)
from spatify.schema import INTEGER

log = logging.getLogger(__name__)


def collapse_tracks(batch: pd.DataFrame) -> pd.DataFrame:
    """Merge rows that share a track_id into one row per track, listing all its genres."""
    batch = batch.sort_values(["track_id", "track_genre"], kind="stable")
    grouped = batch.groupby("track_id", sort=False)
    tracks = grouped.first()
    genres = batch.drop_duplicates(["track_id", "track_genre"]).groupby("track_id", sort=False)
    tracks["genres"] = genres["track_genre"].agg(";".join)
    tracks["n_genres"] = genres["track_genre"].size()
    tracks["popularity"] = grouped["popularity"].max()
    return tracks.drop(columns="track_genre").reset_index()


def add_engineered(batch: pd.DataFrame) -> pd.DataFrame:
    """Add key_sin/key_cos (circle of fifths) and duration_min; restore integer columns."""
    out = batch.copy()
    for col in INTEGER:
        out[col] = out[col].astype("int64")
    key = out["key"].to_numpy()
    tonic = np.where(out["mode"].to_numpy() == 0, (key + 3) % 12, key)  # minor -> relative major
    angle = 2 * np.pi * ((7 * tonic) % 12) / 12  # 7 semitones = one step round the circle
    no_key = key < 0
    out["key_sin"] = np.where(no_key, 0.0, np.sin(angle))
    out["key_cos"] = np.where(no_key, 0.0, np.cos(angle))
    out["duration_min"] = out["duration_ms"] / 60_000
    return out


def build(ds: rd.Dataset, columns: list[str]) -> tuple[rd.Dataset, dict]:
    """Collapse, engineer and scale; return the feature table and the scaler statistics."""
    # Hash-partition by track_id so every row of a track lands in the same block, then
    # collapse each block with pandas (batch_size=None hands over whole blocks).
    n_blocks = max(1, int(ray.cluster_resources().get("CPU", 1)))
    tracks = (
        ds.repartition(n_blocks, keys=["track_id"])
        .map_batches(collapse_tracks, batch_format="pandas", batch_size=None)
        .map_batches(add_engineered, batch_format="pandas")
        .materialize()
    )
    scaler = StandardScaler(columns=columns, output_columns=[f"z_{c}" for c in columns])
    scaled = scaler.fit_transform(tracks).materialize()
    stats = {
        "columns": columns,
        "mean": {c: float(scaler.stats_[f"mean({c})"]) for c in columns},
        "std": {c: float(scaler.stats_[f"std({c})"]) for c in columns},
    }
    return scaled, stats


def reference_sample(tracks: rd.Dataset, columns: list[str], n: int, seed: int) -> pd.DataFrame:
    """A seeded sample of unscaled feature values, the baseline for PSI and KS drift checks."""
    df = tracks.select_columns(["track_id", *columns]).to_pandas().sort_values("track_id")
    return df.sample(n=min(n, len(df)), random_state=seed)[columns].reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the feature table.")
    parser.add_argument("--data", type=Path, default=Path("data/validated/tracks"))
    parser.add_argument("--out", type=Path, default=Path("data/features"))
    parser.add_argument("--metrics", type=Path, default=Path("metrics/features.json"))
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_params("features")
    init_ray()

    start = time.perf_counter()
    ds = rd.read_parquet(str(args.data.resolve()))  # absolute: Ray workers have their own cwd
    rows_in = ds.count()
    tracks, stats = build(ds, cfg["columns"])
    n_tracks = tracks.count()
    if len(tracks.unique("track_id")) != n_tracks:
        log.error("Some track_ids appear more than once after collapsing; stopping")
        raise SystemExit(1)

    fresh_dir(args.out)
    tracks.write_parquet(str((args.out / "tracks").resolve()))
    (args.out / "scaler.json").write_text(json.dumps(stats, indent=2) + "\n")
    reference = reference_sample(tracks, cfg["columns"], cfg["reference_rows"], cfg["seed"])
    reference.to_parquet(args.out / "reference.parquet", index=False)
    n_genres = tracks.select_columns(["n_genres"]).to_pandas()["n_genres"]
    seconds = time.perf_counter() - start

    metrics = {
        "tracks": n_tracks,
        "multi_genre_tracks": int((n_genres > 1).sum()),
        "reference_rows": len(reference),
        **throughput(rows_in, seconds),
    }
    write_metrics(args.metrics, metrics)
    log.info(
        "%d rows -> %d tracks (%d in several genres) in %.1f s (%.0f rows/s)",
        rows_in,
        n_tracks,
        metrics["multi_genre_tracks"],
        metrics["seconds"],
        metrics["rows_per_s"],
    )


if __name__ == "__main__":
    main()
