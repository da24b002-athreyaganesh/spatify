"""Exploratory data analysis of the raw dataset: writes reports/eda/eda.md and its figures.

Covers what the guidelines ask for: nulls, duplicates, genre balance, value ranges and
outliers, and feature distributions. Charts follow the A4 "7 Commandments": colour-blind
safe colours, named axes with units, titles, and a caption under each figure.

Run:  python scripts/eda.py --raw data/raw/spotify_tracks.csv --out reports/eda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

AUDIO = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
NUMERIC = ["popularity", "duration_ms", *AUDIO, "key", "mode", "time_signature"]
UNITS = {"loudness": "dB", "tempo": "BPM", "duration_ms": "ms", "popularity": "0-100 score"}
COLOUR = "#0072B2"  # Okabe-Ito blue, readable for colour-blind viewers


def table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table."""
    cells = df.astype(str).to_numpy().tolist()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in cells])


def label(col: str) -> str:
    return f"{col} ({UNITS[col]})" if col in UNITS else f"{col} (0-1)"


def histograms(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, col in zip(axes.flat, AUDIO, strict=True):
        ax.hist(df[col].dropna(), bins=50, color=COLOUR)
        ax.set_xlabel(label(col))
        ax.set_ylabel("tracks (linear)")
        ax.set_title(col)
    fig.suptitle("Distribution of each audio feature (114,000 rows)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def boxplots(df: pd.DataFrame, path: Path) -> None:
    unit_cols = [c for c in AUDIO if c not in UNITS]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot([df[c].dropna() for c in unit_cols], tick_labels=unit_cols, flierprops={"ms": 2})
    ax.set_ylabel("value (0-1, linear)")
    ax.set_xlabel("audio feature")
    ax.set_title("Spread and outliers of the 0-1 audio features (points beyond 1.5 IQR)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def correlations(df: pd.DataFrame, path: Path) -> None:
    corr = df[NUMERIC].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(corr, cmap="cividis", vmin=-1, vmax=1)  # perceptually uniform, CVD-safe
    ax.set_xticks(range(len(NUMERIC)), NUMERIC, rotation=90)
    ax.set_yticks(range(len(NUMERIC)), NUMERIC)
    fig.colorbar(image, ax=ax, label="Pearson correlation (-1 to 1)")
    ax.set_title("Correlation between numeric columns")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def genres_per_track(per_track: pd.Series, path: Path) -> None:
    counts = per_track.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=COLOUR)
    ax.set_yscale("log")
    ax.set_xlabel("number of genres a track is listed under")
    ax.set_ylabel("tracks (log scale)")
    ax.set_title("Most tracks have one genre; 16,299 have several")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def outlier_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per numeric column: range, mean, zeros and values beyond 1.5 IQR."""
    rows = []
    for col in NUMERIC:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        outside = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        rows.append(
            [col, round(s.min(), 3), round(s.max(), 3), round(s.mean(), 3), (s == 0).sum(), outside]
        )
    return pd.DataFrame(rows, columns=["column", "min", "max", "mean", "zeros", "beyond 1.5 IQR"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="EDA report for the raw dataset.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/spotify_tracks.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/eda"))
    args = parser.parse_args(argv)
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    per_track = df.groupby("track_id")["track_genre"].nunique()
    conflicts = df.groupby("track_id")[NUMERIC].nunique()
    conflicting = ", ".join(
        f"{c} ({(conflicts[c] > 1).sum():,} tracks)" for c in NUMERIC if (conflicts[c] > 1).any()
    )
    corr = df[NUMERIC].corr()
    silent = (df["tempo"] == 0) & (df["time_signature"] == 0)
    genre_rows = df["track_genre"].value_counts()
    nulls = df.isna().sum()

    histograms(df, figures / "histograms.png")
    boxplots(df, figures / "boxplots.png")
    correlations(df, figures / "correlations.png")
    genres_per_track(per_track, figures / "genres_per_track.png")

    null_table = nulls[nulls > 0].rename_axis("column").reset_index(name="missing")
    report = f"""# EDA: Spotify Tracks Dataset

Generated by `scripts/eda.py` from `{args.raw}`. Regenerate with `dvc repro eda`.

## Overview

- {len(df):,} rows, {df.shape[1]} columns (after dropping the unnamed row index).
- {df["track_id"].nunique():,} unique track IDs in {df["track_genre"].nunique()} genres.
- Rows per genre: {genre_rows.min():,} to {genre_rows.max():,}; the genre balance is artificial.

## Missing values

{table(null_table) if len(null_table) else "None."}

## Duplicates

- Exact duplicate rows: {df.duplicated().sum():,}.
- Tracks listed under more than one genre: {(per_track > 1).sum():,}
  (up to {per_track.max()} genres for one track).
- Columns whose value differs between rows of the same track: {conflicting or "none"}.
  The audio features never differ, so the features stage merges these rows into one
  track and keeps the highest popularity.

![Genres per track](figures/genres_per_track.png)

*How many genres each track is listed under. Log scale: most tracks have one.*

## Ranges, zeros and outliers

{table(outlier_counts(df))}

- `tempo` and `time_signature` are 0 together in {silent.sum()} rows,
  {(df.loc[silent, "track_genre"] == "sleep").sum()} of them in the "sleep"
  genre: silent or ambient tracks. The validate stage keeps them with a warning.
- `duration_ms` is 0 in {(df.duration_ms == 0).sum()} row; validate rejects it (the
  shortest real track is {df.loc[df.duration_ms > 0, "duration_ms"].min() / 1000:.1f} s).
- "Beyond 1.5 IQR" counts are high for skewed features (speechiness, instrumentalness,
  liveness). These are real values, not errors, so they are kept; scaling handles them.

![Box plots](figures/boxplots.png)

*Spread of the 0-1 audio features. Boxes span the middle 50%; points are beyond 1.5 IQR.*

## Distributions

![Histograms](figures/histograms.png)

*Each audio feature over all rows. Linear counts; loudness in dB, tempo in BPM.*

![Correlations](figures/correlations.png)

*Pearson correlation between numeric columns. Energy and loudness move together
({corr.loc["energy", "loudness"]:+.2f}), and both move against acousticness
({corr.loc["energy", "acousticness"]:+.2f} and {corr.loc["loudness", "acousticness"]:+.2f}),
so a distance over all three counts that signal more than once; Phase 2 compares
feature subsets.*
"""
    (args.out / "eda.md").write_text(report)
    print(f"Wrote {args.out / 'eda.md'} and 4 figures")


if __name__ == "__main__":
    main()
