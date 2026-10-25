"""Build the base catalogue and simulated incoming batches from the raw dataset.

The base set is what the first model trains on. The batches are dropped into
data/incoming/ later to drive the demo:

- normal:  unseen tracks from the base genres (no drift expected)
- shifted: tracks from held-out genres only (genre-mix drift)
- drifted: unseen base-genre tracks with an additive feature shift, a stand-in for
           the problem statement's "era" shift (the dataset has no release year)

A seeded share of rows in every batch is corrupted (nulls, out-of-range values) so
the validate stage has known bad rows to reject. Everything is driven by the
`batches` section of params.yaml.

Run:  python -m spatify.batches --params params.yaml --raw data/raw/spotify_tracks.csv --out data
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

# Values outside each feature's valid range, used to plant out-of-range rows.
OUT_OF_RANGE = {"danceability": 1.5, "energy": -0.2, "valence": 1.3, "tempo": -1.0}
# Required fields that get blanked to plant null rows.
NULLABLE = ["track_id", "danceability", "energy", "tempo"]


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV, dropping the unnamed row-index column it ships with."""
    df = pd.read_csv(path)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def pick_held_out(genres: list[str], spec: int | list[str], rng: np.random.Generator) -> list[str]:
    """Return the held-out genres: an explicit list, or `spec` genres drawn at random."""
    if isinstance(spec, list):
        missing = set(spec) - set(genres)
        if missing:
            raise ValueError(f"Unknown held-out genres: {sorted(missing)}")
        return sorted(spec)
    return sorted(rng.choice(genres, size=spec, replace=False).tolist())


def split_catalogue(
    df: pd.DataFrame, held_out: list[str], base_fraction: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into the base set, a reserve of unseen base-genre rows, and a shifted pool.

    The split is by track_id, so a track never appears in both the base set and a batch.
    Tracks listed under any held-out genre are kept out of the base set entirely.
    """
    in_held_out = df["track_genre"].isin(held_out)
    held_out_ids = df.loc[in_held_out, "track_id"]
    shifted_pool = df[in_held_out]
    in_dist = df[~in_held_out & ~df["track_id"].isin(held_out_ids)]

    ids = rng.permutation(in_dist["track_id"].drop_duplicates().to_numpy())
    is_base = in_dist["track_id"].isin(ids[: int(len(ids) * base_fraction)])
    return in_dist[is_base], in_dist[~is_base], shifted_pool


def shift_features(
    batch: pd.DataFrame, shifts: dict[str, float], bounds: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Add a fixed offset to each feature, clipped to the range seen in the raw data."""
    batch = batch.copy()
    for col, delta in shifts.items():
        low, high = bounds[col]
        batch[col] = (batch[col] + delta).clip(low, high)
    return batch


def corrupt(
    batch: pd.DataFrame, fraction: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Plant nulls and out-of-range values in a seeded share of rows; return the counts."""
    batch = batch.copy()
    n_bad = round(len(batch) * fraction)
    n_null = n_bad // 2
    rows = rng.choice(len(batch), size=n_bad, replace=False)
    out_cols = list(OUT_OF_RANGE)
    for i, row in enumerate(rows):
        if i < n_null:
            col = NULLABLE[i % len(NULLABLE)]
            batch.iloc[row, batch.columns.get_loc(col)] = np.nan
        else:
            col = out_cols[i % len(out_cols)]
            batch.iloc[row, batch.columns.get_loc(col)] = OUT_OF_RANGE[col]
    return batch, {"null": n_null, "out_of_range": n_bad - n_null}


def build(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Return the base set, the named batches, and a manifest describing them."""
    rng = np.random.default_rng(cfg["seed"])
    held_out = pick_held_out(sorted(df["track_genre"].unique()), cfg["held_out_genres"], rng)
    base, reserve, shifted_pool = split_catalogue(df, held_out, cfg["base_fraction"], rng)

    size = cfg["batch_size"]
    plan = [
        ("normal", cfg["n_normal"], "reserve"),
        ("drifted", cfg["n_drifted"], "reserve"),
        ("shifted", cfg["n_shifted"], "shifted"),
    ]
    pools = {
        "reserve": reserve.sample(frac=1, random_state=rng),
        "shifted": shifted_pool.sample(frac=1, random_state=rng),
    }
    for name, pool in pools.items():
        needed = size * sum(count for _, count, pool_name in plan if pool_name == name)
        if needed > len(pool):
            raise ValueError(f"The {name} pool has {len(pool)} rows, not enough for {needed}")

    bounds = {col: (df[col].min(), df[col].max()) for col in cfg["drift"]}
    offsets = {"reserve": 0, "shifted": 0}
    batches, entries = {}, []
    for kind, count, pool_name in plan:
        for i in range(1, count + 1):
            start = offsets[pool_name]
            batch = pools[pool_name].iloc[start : start + size]
            offsets[pool_name] += size
            if kind == "drifted":
                batch = shift_features(batch, cfg["drift"], bounds)
            # Corrupt after shifting so the planted values are not clipped away.
            batch, planted = corrupt(batch, cfg["corrupt_fraction"], rng)
            name = f"{kind}_{i:02d}"
            batches[name] = batch
            entries.append(
                {
                    "name": name,
                    "kind": kind,
                    "rows": len(batch),
                    "genres": int(batch["track_genre"].nunique()),
                    "planted": planted,
                }
            )

    manifest = {
        "seed": cfg["seed"],
        "held_out_genres": held_out,
        "base_rows": len(base),
        "base_tracks": int(base["track_id"].nunique()),
        "drift": cfg["drift"],
        "batches": entries,
    }
    return base, batches, manifest


def write(out_dir: Path, base: pd.DataFrame, batches: dict[str, pd.DataFrame], manifest: dict):
    """Write data/base/tracks.csv, one CSV per batch, and data/batches/manifest.json."""
    base_dir, batch_dir = out_dir / "base", out_dir / "batches"
    base_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    base.to_csv(base_dir / "tracks.csv", index=False)
    for name, batch in batches.items():
        batch.to_csv(batch_dir / f"{name}.csv", index=False)
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the base set and simulated batches.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/spotify_tracks.csv"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = yaml.safe_load(args.params.read_text())["batches"]
    df = load_raw(args.raw)
    log.info("Loaded %d rows and %d genres from %s", len(df), df["track_genre"].nunique(), args.raw)

    base, batches, manifest = build(df, cfg)
    write(args.out, base, batches, manifest)
    log.info("Held-out genres: %s", ", ".join(manifest["held_out_genres"]))
    log.info("Wrote the base set (%d rows) and %d batches to %s", len(base), len(batches), args.out)


if __name__ == "__main__":
    main()
