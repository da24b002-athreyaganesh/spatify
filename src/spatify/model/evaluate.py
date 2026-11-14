"""Evaluate stage: score every candidate on held-out seed tracks, in its packaged form.

Each candidate is loaded back from MLflow with mlflow.pyfunc.load_model, the way the model
server will load it, so this also tests the packaging. The seed tracks come from the
held-out split: their genres were never used to fit a model.

Metrics per candidate, logged into the candidate's own training run so that
mlflow.search_runs can rank them (register stage):
- genre_match_at_k: share of the top-K recommendations that share a genre with the seed.
  There is no feedback yet, so genre stands in for "sounds alike"; this is the
  promotion metric.
- genre_match_at_1: the same for the top recommendation alone.
- mean_score_at_k: mean cosine similarity of the top K, in the candidate's own feature
  space, so it isn't comparable across feature sets.
- coverage_at_k: share of the catalogue recommended at least once. Low coverage means a
  few "hub" tracks are recommended to everyone.
- ms_per_query: scoring time, a first look at serving latency.
Random recommendations give a baseline for genre_match_at_k.

Run through MLflow (as DVC does):
    mlflow run . -e evaluate --env-manager=local --experiment-name spatify
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from spatify.model.data import eval_queries, genre_sets, load_tracks
from spatify.model.plots import plt, variants_figure
from spatify.model.tracking import lineage
from spatify.pipeline.common import load_params, setup_logging

log = logging.getLogger(__name__)

METRIC = "genre_match_at_k"  # the promotion metric


def score(
    predict: Callable[[pd.DataFrame, dict], pd.DataFrame],
    query_ids: list[str],
    genres: dict[str, set[str]],
    k: int,
    catalogue_size: int,
) -> dict[str, float]:
    """Metrics for one model, given its predict function and the seed tracks."""
    start = time.perf_counter()
    recs = predict(pd.DataFrame({"track_id": query_ids}), {"k": k})
    seconds = time.perf_counter() - start
    seed_genres = recs["query"].map(lambda q: genres[query_ids[q]])
    rec_genres = recs["genres"].str.split(";").map(set)
    recs = recs.assign(match=[bool(a & b) for a, b in zip(seed_genres, rec_genres, strict=True)])
    return {
        "genre_match_at_k": round(float(recs.groupby("query")["match"].mean().mean()), 4),
        "genre_match_at_1": round(float(recs.loc[recs["rank"] == 1, "match"].mean()), 4),
        "mean_score_at_k": round(float(recs["score"].mean()), 4),
        "coverage_at_k": round(recs["track_id"].nunique() / catalogue_size, 4),
        "ms_per_query": round(1000 * seconds / len(query_ids), 3),
    }


def random_baseline(
    tracks: pd.DataFrame, query_ids: list[str], genres: dict[str, set[str]], k: int, seed: int
) -> float:
    """genre_match_at_k for k random tracks per seed: the score of a model that knows nothing."""
    rng = np.random.default_rng(seed)
    ids = tracks["track_id"].to_numpy()
    track_genres = tracks["genres"].str.split(";").map(set).to_numpy()
    shares = []
    for query in query_ids:
        picks = [p for p in rng.choice(len(ids), size=k + 1, replace=False) if ids[p] != query]
        shares.append(np.mean([bool(genres[query] & track_genres[p]) for p in picks[:k]]))
    return round(float(np.mean(shares)), 4)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the candidate models.")
    parser.add_argument("--data", type=Path, default=Path("data/features/tracks"))
    parser.add_argument("--candidates", type=Path, default=Path("models/candidates.json"))
    parser.add_argument("--metrics", type=Path, default=Path("metrics/evaluate.json"))
    parser.add_argument("--plots", type=Path, default=Path("reports/evaluate"))
    args = parser.parse_args(argv)
    setup_logging()
    split, cfg = load_params("split"), load_params("evaluate")
    k = cfg["k"]

    tracks = load_tracks(args.data)
    genres = genre_sets(tracks)
    query_ids = eval_queries(tracks, split, cfg["queries"], cfg["seed"])
    candidates = json.loads(args.candidates.read_text())
    client = MlflowClient()
    rows = []
    with mlflow.start_run():  # under `mlflow run`, this is the run it started
        mlflow.set_tags(
            {**lineage(), "stage": "evaluate", "train_run_id": candidates["train_run_id"]}
        )
        mlflow.log_params({"k": k, "queries": len(query_ids), "query_seed": cfg["seed"]})
        for cand in candidates["candidates"]:
            model = mlflow.pyfunc.load_model(cand["model_uri"])
            predict = lambda df, params, m=model: m.predict(df, params=params)  # noqa: E731
            metrics = score(predict, query_ids, genres, k, len(tracks))
            for name, value in metrics.items():
                client.log_metric(cand["run_id"], name, value)  # into the candidate's run
            mlflow.log_metrics({f"{cand['variant']}.{n}": v for n, v in metrics.items()})
            rows.append({key: cand[key] for key in ("variant", "feature_set", "weighting")})
            rows[-1].update(metrics)
            log.info("%s: genre match@%d %.4f", cand["variant"], k, metrics[METRIC])
        baseline = random_baseline(tracks, query_ids, genres, k, cfg["seed"])
        mlflow.log_metric("random_baseline.genre_match_at_k", baseline)

        table = pd.DataFrame(rows).sort_values(METRIC, ascending=False, kind="stable")
        args.plots.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.plots / "variants.csv", index=False)
        fig = variants_figure(table, baseline, k, len(query_ids))
        fig.savefig(args.plots / "variants.png", dpi=110)
        mlflow.log_figure(fig, "variants.png")
        plt.close(fig)
        mlflow.log_table(table, "variants.json")

    best = table.iloc[0]
    summary = {
        "k": k,
        "queries": len(query_ids),
        "random_baseline": baseline,
        "best_variant": best["variant"],
        "best": {name: float(best[name]) for name in metrics},
        "variants": {
            row["variant"]: {name: float(row[name]) for name in metrics}
            for _, row in table.iterrows()
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(summary, indent=2) + "\n")
    log.info(
        "Best: %s, genre match@%d %.4f (random tracks: %.4f)",
        best["variant"],
        k,
        best[METRIC],
        baseline,
    )


if __name__ == "__main__":
    main()
