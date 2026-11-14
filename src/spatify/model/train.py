"""Train stage: build every candidate model and log each one to MLflow as a nested run.

Each candidate is one feature set x one weighting (params.yaml `train`). Building a
nearest-neighbour model needs no training loop: it scales the catalogue with the features
stage's scaler, applies the weights, and packages everything as a pyfunc model. Fisher
weights are fitted on the training split only; held-out tracks' genres stay unseen.

The parent run records the lineage shared by all candidates: git commit, DVC data
version, trigger, environment, the Phase 1 data-quality metrics and the training feature
distributions (the drift baseline). The candidates go to models/candidates.json for the
evaluate stage.

Run through MLflow (as DVC does):
    mlflow run . -e train --env-manager=local --experiment-name spatify
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from spatify.model.data import eval_mask, load_tracks
from spatify.model.plots import plt, weights_figure
from spatify.model.pyfunc import log_candidate
from spatify.model.recommender import DETAILS, Recommender, fisher_weights
from spatify.model.tracking import environment, lineage
from spatify.pipeline.common import load_params, setup_logging

log = logging.getLogger(__name__)


def variants(cfg: dict, columns: list[str]) -> Iterator[tuple[str, str, str, list[str]]]:
    """(name, feature set, weighting, feature columns) for every candidate."""
    for feature_set, dropped in cfg["feature_sets"].items():
        kept = [c for c in columns if c not in dropped]
        for weighting in cfg["weightings"]:
            yield f"{feature_set}-{weighting}", feature_set, weighting, kept


def build(
    tracks: pd.DataFrame, columns: list[str], scaler: dict, weighting: str, fit: np.ndarray
) -> Recommender:
    """One candidate. `fit` marks the training split, the only tracks weights learn from."""
    mean = np.array([scaler["mean"][c] for c in columns])
    std = np.array([scaler["std"][c] for c in columns])
    if weighting == "uniform":
        weights = np.ones(len(columns))
    elif weighting == "fisher":
        z = (tracks.loc[fit, columns].to_numpy(dtype=np.float64) - mean) / std
        weights = fisher_weights(z, tracks.loc[fit, "genres"])
    else:
        raise ValueError(f"Unknown weighting {weighting!r}; use uniform or fisher")
    return Recommender(tracks[DETAILS + columns], columns, mean, std, weights)


def distributions(reference: pd.DataFrame) -> dict:
    """Summary of the training feature distributions (the drift-check baseline)."""
    quantiles = reference.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        col: {
            "mean": float(reference[col].mean()),
            "std": float(reference[col].std()),
            **{f"p{int(q * 100)}": float(v) for q, v in quantiles[col].items()},
        }
        for col in reference.columns
    }


def log_lineage(features_dir: Path, tracks: pd.DataFrame, fit: np.ndarray, tags: dict) -> None:
    """Parent-run records shared by every candidate."""
    mlflow.set_tags({**tags, "stage": "train"})
    mlflow.log_params(
        {
            "n_tracks": len(tracks),
            "n_fit_tracks": int(fit.sum()),
            "n_eval_tracks": int((~fit).sum()),
        }
    )
    mlflow.log_dict(environment(), "environment.json")
    reference = pd.read_parquet(features_dir / "reference.parquet")
    mlflow.log_dict(distributions(reference), "feature_distributions.json")
    mlflow.log_artifact(str(features_dir / "reference.parquet"), "drift")
    mlflow.log_artifact(str(features_dir / "scaler.json"))
    for path in ["params.yaml", "python_env.yaml"]:
        if Path(path).exists():
            mlflow.log_artifact(path)
    # Phase 1's data-quality metrics, so a model links to the state of the data it saw.
    for stage in ["ingest", "validate", "features"]:
        path = Path(f"metrics/{stage}.json")
        if path.exists():
            mlflow.log_artifact(str(path), "data_metrics")
            values = json.loads(path.read_text())
            for name in ["rows", "rejected", "reject_rate", "tracks", "rows_per_s"]:
                if isinstance(values.get(name), int | float):
                    mlflow.log_metric(f"data.{stage}.{name}", values[name])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and log the candidate models.")
    parser.add_argument("--data", type=Path, default=Path("data/features"))
    parser.add_argument("--out", type=Path, default=Path("models/candidates.json"))
    args = parser.parse_args(argv)
    setup_logging()
    columns = load_params("features")["columns"]
    split, cfg = load_params("split"), load_params("train")

    tracks = load_tracks(args.data / "tracks")
    scaler = json.loads((args.data / "scaler.json").read_text())
    fit = ~eval_mask(tracks["track_id"], split["eval_fraction"], split["seed"])
    tags = lineage()
    candidates = []
    with mlflow.start_run() as parent:  # under `mlflow run`, this is the run it started
        log_lineage(args.data, tracks, fit, tags)
        mlflow.log_params({"split_seed": split["seed"], "eval_fraction": split["eval_fraction"]})
        for name, feature_set, weighting, cols in variants(cfg, columns):
            recommender = build(tracks, cols, scaler, weighting, fit)
            with mlflow.start_run(run_name=name, nested=True) as run:
                mlflow.set_tags({**tags, "stage": "train", "variant": name})
                mlflow.log_params(
                    {
                        "variant": name,
                        "feature_set": feature_set,
                        "weighting": weighting,
                        "n_features": len(cols),
                        "features": ",".join(cols),
                        "k": cfg["k"],
                        "split_seed": split["seed"],
                        "eval_fraction": split["eval_fraction"],
                    }
                )
                weights = dict(zip(cols, recommender.weights.round(4).tolist(), strict=True))
                mlflow.log_dict(weights, "weights.json")
                if weighting != "uniform":
                    fig = weights_figure(cols, recommender.weights, name)
                    mlflow.log_figure(fig, "weights.png")
                    plt.close(fig)
                info = {
                    "run_id": run.info.run_id,
                    "variant": name,
                    "dataset_version": tags["dataset_version"],
                }
                model_info = log_candidate(recommender, info, cfg["k"])
            candidates.append(
                {
                    "variant": name,
                    "feature_set": feature_set,
                    "weighting": weighting,
                    "run_id": run.info.run_id,
                    "model_uri": model_info.model_uri,
                }
            )
            log.info("Logged candidate %s (run %s)", name, run.info.run_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_id": parent.info.experiment_id,
        "train_run_id": parent.info.run_id,
        "candidates": candidates,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("%d candidates logged under run %s", len(candidates), parent.info.run_id)


if __name__ == "__main__":
    main()
