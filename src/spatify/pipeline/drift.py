"""Drift check stage: how far each incoming batch has moved from the training data (D35).

The reference is the base catalogue's validated rows (source == params drift.reference):
the data every model is trained on. For every other source (a batch file from
data/incoming) and every audio feature it computes:
- the Population Stability Index (PSI) over bins cut at the reference's deciles;
- the two-sample Kolmogorov-Smirnov (KS) statistic and p-value.

A feature is "major" at PSI >= psi_major (0.2), "moderate" at >= psi_moderate (0.1),
otherwise "stable". A batch has drifted when any feature is major, or at least
min_moderate features are moderate. KS is reported but doesn't decide: against ~80,000
reference rows it calls even normal batches different.

Writes metrics/drift.json, which the worker and the gateway's /drift read.
Run:  python -m spatify.pipeline.drift
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from spatify.pipeline.common import load_params, write_metrics

log = logging.getLogger(__name__)

FLOOR = 1e-4  # share given to an empty bin, so the log in PSI stays finite


def psi(reference: np.ndarray, new: np.ndarray, bins: int) -> float:
    """PSI = sum over bins of (new% - ref%) * ln(new% / ref%), bins at reference quantiles.

    Quantile bins hold about the same share of the reference each; repeated edges (a
    feature that is often exactly 0) are merged, so such a feature gets fewer bins.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), FLOOR, None)
    cur = np.clip(np.histogram(new, edges)[0] / len(new), FLOOR, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def level(value: float, cfg: dict) -> str:
    if value >= cfg["psi_major"]:
        return "major"
    return "moderate" if value >= cfg["psi_moderate"] else "stable"


def batch_drift(reference: pd.DataFrame, batch: pd.DataFrame, cfg: dict) -> dict:
    """Per-feature PSI and KS for one batch, and whether the batch has drifted."""
    features = {}
    for name in cfg["features"]:
        ref, new = reference[name].dropna().to_numpy(), batch[name].dropna().to_numpy()
        value = psi(ref, new, cfg["bins"])
        ks = ks_2samp(ref, new)
        features[name] = {
            "psi": round(value, 4),
            "ks_stat": round(float(ks.statistic), 4),
            "ks_pvalue": float(f"{ks.pvalue:.3g}"),
            "level": level(value, cfg),
        }
    levels = [f["level"] for f in features.values()]
    drifted = "major" in levels or levels.count("moderate") >= cfg["min_moderate"]
    return {"rows": len(batch), "drifted": drifted, "features": features}


def report(tracks: pd.DataFrame, cfg: dict) -> dict:
    """The drift report for every non-reference source in the validated rows."""
    reference = tracks[tracks["source"] == cfg["reference"]]
    if reference.empty:
        raise SystemExit(f"No reference rows (source {cfg['reference']!r}) in the validated data")
    batches = {
        source: batch_drift(reference, rows, cfg)
        for source, rows in sorted(tracks[tracks["source"] != cfg["reference"]].groupby("source"))
    }
    return {
        "reference": cfg["reference"],
        "reference_rows": len(reference),
        "thresholds": {k: cfg[k] for k in ("psi_moderate", "psi_major", "min_moderate")},
        "drifted_batches": [s for s, b in batches.items() if b["drifted"]],
        "batches": batches,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check incoming batches for drift.")
    parser.add_argument("--data", type=Path, default=Path("data/validated/tracks"))
    parser.add_argument("--metrics", type=Path, default=Path("metrics/drift.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_params("drift")

    tracks = pd.read_parquet(args.data, columns=["source", *cfg["features"]])
    result = report(tracks, cfg)
    write_metrics(args.metrics, result)
    for source, batch in result["batches"].items():
        worst = max(batch["features"].items(), key=lambda kv: kv[1]["psi"])
        log.info(
            "%s: %s (largest PSI %.3f, %s)",
            source,
            "DRIFTED" if batch["drifted"] else "no drift",
            worst[1]["psi"],
            worst[0],
        )
    if not result["batches"]:
        log.info("No incoming batches to check")


if __name__ == "__main__":
    main()
