"""Offline check of learning from feedback, on the real catalogue (D37, D38, D39).

No stack needed. For each seed:
1. Simulated listeners rate the serving model's lists (all features; `--serving` picks
   uniform or Fisher weights), served as the gateway serves them: top 9 and one random pick.
2. The feedback model is fitted on the earlier lists, as the train stage does, and the
   models are judged on the latest ones (NDCG@k and hit rate@k, as the evaluate stage and
   the promotion gate do). These lists are the serving model's picks (off-policy, D37).
3. Fresh listeners rate each model's own lists: the like rate a deployment would see, also
   for a forced-bad copy of the serving model (scripts/force_bad_model.py).

Run from spatify/ after `dvc repro features` (it reads data/features):
    uv run python scripts/feedback_experiment.py --lists 200 --seeds 0 1 2
    uv run python scripts/feedback_experiment.py --serving fisher --lists 400   # like v1
200 lists are about 2,000 likes and skips, when the worker's first feedback retrain runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from force_bad_model import scrambled  # scripts/ is on the path when run as a script

from spatify.model.data import eval_mask, load_tracks
from spatify.model.feedback import (
    fit_weights,
    random_ranking_metrics,
    ranking_metrics,
    split_by_time,
)
from spatify.model.recommender import DETAILS, Recommender
from spatify.model.train import build
from spatify.pipeline.common import load_params
from spatify.simulation import TASTE, offline_lists


def like_rate(events) -> float:
    """Likes per rating on the model's own picks (exploration slots left out)."""
    return float((events.loc[~events["explored"], "action"] == "like").mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline check of learning from feedback.")
    parser.add_argument("--lists", type=int, default=200, help="lists to learn from, per seed")
    parser.add_argument("--live", type=int, default=150, help="lists per like-rate check")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--data", type=Path, default=Path("data/features"))
    parser.add_argument(
        "--serving", choices=["uniform", "fisher"], default="uniform", help="the champion"
    )
    args = parser.parse_args()
    columns = load_params("features")["columns"]
    split, fb = load_params("split"), load_params("feedback")
    tracks = load_tracks(args.data / "tracks")
    scaler = json.loads((args.data / "scaler.json").read_text())
    fit = ~eval_mask(tracks["track_id"], split["eval_fraction"], split["seed"])
    uniform = build(tracks, columns, scaler, "uniform", fit)  # fit_weights' base
    serving = (
        uniform if args.serving == "uniform" else build(tracks, columns, scaler, "fisher", fit)
    )
    bad = scrambled(serving, seed=0)
    sounds = tracks.set_index("track_id")[list(TASTE)]
    k, label = fb["k"], f"{args.serving} (serving)"
    print(
        f"{len(tracks):,} tracks; {args.lists} lists per seed, served by all-{args.serving}; "
        f"NDCG and hit rate at {k}"
    )

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        events = offline_lists(serving, sounds, args.lists, rng)
        train, test = split_by_time(events, fb["test_fraction"])
        weights, report = fit_weights(uniform, train, fb["min_events"], fb["c"])
        print(f"\n== seed {seed}: {len(events):,} ratings, {report['events']:,} to learn from")
        if weights is None:
            print("   no feedback model (too few usable ratings)")
            continue
        learned = Recommender(
            tracks[DETAILS + columns], columns, uniform.mean, uniform.std, weights
        )
        top = sorted(zip(columns, weights, strict=True), key=lambda cw: -cw[1])
        print("   learned weights: " + ", ".join(f"{c} {w:.2f}" for c, w in top))
        judged = [(label, serving), ("feedback", learned)]
        if serving is not uniform:
            judged.append(("uniform", uniform))
        for name, result in (
            *((n, ranking_metrics(m, test, k)) for n, m in judged),
            ("random order", random_ranking_metrics(test, k, seed)),
        ):
            scores = f"NDCG {result['ndcg']:.3f}  hit rate {result['hit_rate']:.3f}"
            lists = f" on {result['lists']} held-out lists" if "lists" in result else ""
            print(f"   {name:16} {scores}{lists}")
        live = {}  # each model's lists, rated by the same fresh listeners
        for name, model in ((label, serving), ("feedback", learned), ("forced bad", bad)):
            rng = np.random.default_rng(seed + 100)
            live[name] = like_rate(offline_lists(model, sounds, args.live, rng))
        print("   like rate: " + ", ".join(f"{n} {100 * r:.1f}%" for n, r in live.items()))


if __name__ == "__main__":
    main()
