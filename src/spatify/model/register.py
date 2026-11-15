"""Register stage: register the best candidate; promote it only if it beats the champion.

The promotion gate from L10 (Task A):
1. Pick this training run's best candidate with mlflow.search_runs on genre_match_at_k.
2. Register it as a new version of the model, tagged with its dataset version, git
   commit, trigger and variant (L3), and point the `challenger` alias at it.
3. Score the challenger and the current champion on the same held-out seed tracks:
   tracks neither model was fitted on, and only ones in both models' catalogues.
4. Promote when the challenger wins by at least `min_gain`. `previous` moves to the old
   champion (the rollback target) and `champion` to the new version. With no champion
   yet, the challenger is promoted directly.
Not promoting is a normal outcome, not a failure: the champion keeps serving.

The metric is genre match@K until there is enough held-out feedback; then it is NDCG@k on
the held-out likes and skips, with its own min_gain (the evaluate stage decides, D37).

Run through MLflow (as DVC does):
    mlflow run . -e register --env-manager=local --experiment-name spatify
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from spatify.model.data import eval_queries, genre_sets, load_tracks
from spatify.model.evaluate import FEEDBACK_METRIC, METRIC, score
from spatify.model.feedback import load_events, ranking_metrics, split_by_time
from spatify.model.tracking import lineage
from spatify.pipeline.common import load_params, setup_logging

log = logging.getLogger(__name__)


def should_promote(challenger: float, champion: float | None, min_gain: float) -> bool:
    """The gate: promote with no champion, or when the challenger wins by min_gain or more."""
    return champion is None or challenger - champion >= min_gain


def alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    """The version an alias points to, or None if the alias (or the model) doesn't exist."""
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def promote(client: MlflowClient, name: str, version: str, champion: str | None) -> None:
    """Move the aliases: previous -> old champion, champion -> version; drop challenger."""
    if champion is not None:
        client.set_registered_model_alias(name, "previous", champion)
    client.set_registered_model_alias(name, "champion", version)
    if alias_version(client, name, "challenger") is not None:
        client.delete_registered_model_alias(name, "challenger")


def best_candidate(candidates: dict, metric: str = METRIC) -> dict:
    """The candidate with the highest `metric` (ties: higher genre match@1)."""
    runs = mlflow.search_runs(
        experiment_ids=[candidates["experiment_id"]],
        filter_string=f"tags.mlflow.parentRunId = '{candidates['train_run_id']}'",
        order_by=[f"metrics.{metric} DESC", "metrics.genre_match_at_1 DESC"],
        max_results=1,
    )
    if runs.empty or runs.get(f"metrics.{metric}", pd.Series(dtype=float)).isna().all():
        raise SystemExit("No evaluated candidates for this training run; run evaluate first")
    run_id = runs.iloc[0]["run_id"]
    return next(c for c in candidates["candidates"] if c["run_id"] == run_id)


def gate_scores(
    models: dict[str, mlflow.pyfunc.PyFuncModel],
    query_ids: list[str],
    genres: dict[str, set[str]],
    k: int,
) -> tuple[dict[str, float], int]:
    """genre_match_at_k per model on the seed tracks every model knows, and how many."""
    catalogues = {
        role: model.unwrap_python_model().recommender.positions for role, model in models.items()
    }
    common = [q for q in query_ids if all(q in ids for ids in catalogues.values())]
    scores = {}
    for role, model in models.items():
        predict = lambda df, params, m=model: m.predict(df, params=params)  # noqa: E731
        scores[role] = score(predict, common, genres, k, len(catalogues[role]))[METRIC]
    return scores, len(common)


def feedback_gate_scores(
    models: dict[str, mlflow.pyfunc.PyFuncModel], test_events: pd.DataFrame, k: int
) -> tuple[dict[str, float], int]:
    """NDCG@k per model on the held-out lists, and how many lists the challenger was judged on."""
    results = {
        role: ranking_metrics(model.unwrap_python_model().recommender, test_events, k)
        for role, model in models.items()
    }
    return {role: r["ndcg"] for role, r in results.items()}, results["challenger"]["lists"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Register the best candidate; maybe promote.")
    parser.add_argument("--data", type=Path, default=Path("data/features/tracks"))
    parser.add_argument("--candidates", type=Path, default=Path("models/candidates.json"))
    parser.add_argument("--evaluation", type=Path, default=Path("metrics/evaluate.json"))
    parser.add_argument("--metrics", type=Path, default=Path("metrics/register.json"))
    args = parser.parse_args(argv)
    setup_logging()
    split, ev, cfg = load_params("split"), load_params("evaluate"), load_params("register")
    fb = load_params("feedback")
    name, k = cfg["model_name"], ev["k"]
    evaluation = json.loads(args.evaluation.read_text())
    metric = evaluation.get("feedback", {}).get("promotion_metric", METRIC)
    min_gain = cfg["min_gain_feedback"] if metric == FEEDBACK_METRIC else cfg["min_gain"]

    candidates = json.loads(args.candidates.read_text())
    best = best_candidate(candidates, metric)
    tracks = load_tracks(args.data)
    query_ids = eval_queries(tracks, split, ev["queries"], ev["seed"])
    client = MlflowClient()
    tags = lineage()
    with mlflow.start_run():  # under `mlflow run`, this is the run it started
        mlflow.set_tags(
            {
                **tags,
                "stage": "register",
                "train_run_id": candidates["train_run_id"],
                "candidate_run_id": best["run_id"],
                "variant": best["variant"],
            }
        )
        version = str(mlflow.register_model(best["model_uri"], name).version)
        version_tags = {
            **tags,
            "variant": best["variant"],
            "train_run_id": candidates["train_run_id"],
            "candidate_run_id": best["run_id"],
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(name, version, key, value)
        client.set_registered_model_alias(name, "challenger", version)

        champion = alias_version(client, name, "champion")
        models = {"challenger": mlflow.pyfunc.load_model(f"models:/{name}/{version}")}
        if champion is not None:
            models["champion"] = mlflow.pyfunc.load_model(f"models:/{name}@champion")
        if metric == FEEDBACK_METRIC:
            _, test_events = split_by_time(load_events(Path(fb["path"])), fb["test_fraction"])
            scores, n_queries = feedback_gate_scores(models, test_events, fb["k"])
            judged_on = f"NDCG@{fb['k']} on {n_queries:,} held-out feedback lists"
        else:
            scores, n_queries = gate_scores(models, query_ids, genre_sets(tracks), k)
            judged_on = f"genre match@{k} on {n_queries:,} held-out seed tracks"
        champion_score = scores.get("champion")
        promoted = should_promote(scores["challenger"], champion_score, min_gain)
        if promoted:
            promote(client, name, version, champion)
        gain = None if champion_score is None else round(scores["challenger"] - champion_score, 4)

        client.update_model_version(
            name,
            version,
            description=(
                f"{best['variant']}: {scores['challenger']:.4f} {judged_on}; "
                + ("promoted to champion" if promoted else f"not promoted (gain {gain})")
            ),
        )
        mlflow.log_params(
            {"metric": metric, "min_gain": min_gain, "gate_queries": n_queries, "k": k}
        )
        mlflow.log_metrics(
            {f"{role}.{metric}": value for role, value in scores.items()}
            | ({"gain": gain} if gain is not None else {})
            | {"promoted": int(promoted)}
        )
        mlflow.set_tags({"promoted": str(promoted).lower(), "registered_version": version})

    summary = {
        "model_name": name,
        "version": int(version),
        "variant": best["variant"],
        "metric": metric,
        "promoted": promoted,
        "challenger_score": scores["challenger"],
        "champion_score": champion_score,
        "gain": gain,
        "min_gain": min_gain,
        "gate_queries": n_queries,
        "champion_version": int(alias_version(client, name, "champion")),
        "previous_version": (
            int(v) if (v := alias_version(client, name, "previous")) is not None else None
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(summary, indent=2) + "\n")
    log.info(
        "Registered %s version %s (%s): %s",
        name,
        version,
        best["variant"],
        "promoted to champion"
        if promoted
        else f"not promoted on {metric}, gain {gain} < {min_gain}; version "
        f"{summary['champion_version']} stays champion",
    )


if __name__ == "__main__":
    main()
