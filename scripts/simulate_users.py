"""Simulated listeners, through the gateway: feedback for retraining, and load (D38).

Each list is a new listener with hidden preferences (spatify.simulation): they ask for a
seed song or two or three sliders, get recommendations from the real gateway, and like or
skip each one through /feedback. Their ears are the local catalogue (data/features/tracks):
the gateway returns no audio features, and the system never sees what listeners care about.

Run with the stack up, from spatify/:
    uv run python scripts/simulate_users.py --lists 200                   # ~2,000 likes and skips
    uv run python scripts/simulate_users.py --lists 2000 --concurrency 8  # load for Grafana

Prints progress, then a summary: the like rate per model version, how often the explored
(random) picks were liked, errors and throughput. The worker retrains once
feedback.retrain_every new likes and skips have arrived (params.yaml).
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from spatify.model.data import load_tracks
from spatify.simulation import TASTE, listener_for, new_query, rate_list


def one_list(api: httpx.Client, i: int, args, sounds: pd.DataFrame, ids: np.ndarray) -> dict:
    """One listener: ask, listen, like or skip each track. Never raises; errors are counted."""
    rng = np.random.default_rng([args.seed, i])  # per list, so threads don't change the run
    query = new_query(rng, ids, args.slider_share)
    out = {"version": None, "like": 0, "skip": 0, "explored": 0, "explored_likes": 0}
    try:
        shown = api.post("/recommend", json={**query, "k": args.k})
        if not shown.is_success:
            return out | {"error": f"/recommend {shown.status_code}: {shown.text[:150]}"}
        body = shown.json()
        out["version"] = body["model_version"]
        recs = {r["track_id"]: r for r in body["recommendations"]}
        listener, target = listener_for(query, sounds, rng)
        for track_id, action in rate_list(listener, target, list(recs), sounds, rng):
            sent = api.post(
                "/feedback",
                json={"request_id": body["request_id"], "track_id": track_id, "action": action},
            )
            if not sent.is_success:  # 201 Created
                return out | {"error": f"/feedback {sent.status_code}: {sent.text[:150]}"}
            out[action] += 1
            if recs[track_id]["explored"]:
                out["explored"] += 1
                out["explored_likes"] += action == "like"
    except httpx.HTTPError as err:
        return out | {"error": f"{type(err).__name__}: {err}"}
    return out


def summary(totals: dict, explored: list[int], errors: int, lists: int, seconds: float) -> str:
    lines = [
        f"Simulated {lists:,} lists in {seconds:.1f} s ({lists / seconds:.1f} lists/s)",
        "model version | lists | likes | skips | like rate",
    ]
    for version, t in sorted(totals.items(), key=lambda item: str(item[0])):
        rated = t["like"] + t["skip"]
        rate = f"{100 * t['like'] / rated:.1f}%" if rated else "—"
        counts = f"{t['lists']:>5} | {t['like']:>5} | {t['skip']:>5}"
        lines.append(f"{version!s:>13} | {counts} | {rate:>9}")
    if explored[1]:
        liked, shown = explored
        share = f"{100 * liked / shown:.0f}%"
        lines.append(f"Random (explored) picks liked: {liked} of {shown} ({share})")
    lines.append(f"Lists that failed: {errors}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated listeners for the running stack.")
    parser.add_argument("--url", default="http://localhost:8000", help="the gateway")
    parser.add_argument("--lists", type=int, default=200, help="listeners, one list each")
    parser.add_argument("--k", type=int, default=10, help="songs per list")
    parser.add_argument("--concurrency", type=int, default=4, help="listeners at a time")
    parser.add_argument("--slider-share", type=float, default=0.3, help="lists from sliders")
    parser.add_argument("--seed", type=int, default=None, help="repeat a run (default: random)")
    parser.add_argument("--data", type=Path, default=Path("data/features/tracks"))
    args = parser.parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**31)

    api = httpx.Client(base_url=args.url, timeout=15)
    try:
        health = api.get("/health")
    except httpx.HTTPError as err:
        sys.exit(f"Can't reach the gateway at {args.url} ({err}); is the stack up?")
    if health.status_code != 200:
        sys.exit(f"The gateway isn't ready: {health.text}")
    tracks = load_tracks(args.data)
    sounds = tracks.set_index("track_id")[list(TASTE)]
    ids = tracks["track_id"].astype(str).to_numpy()
    print(f"Seed {args.seed} (pass --seed {args.seed} to repeat); {len(ids):,} songs to pick from")

    totals = defaultdict(lambda: {"lists": 0, "like": 0, "skip": 0})
    explored, errors, done = [0, 0], 0, 0
    start = time.monotonic()
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(one_list, api, i, args, sounds, ids) for i in range(args.lists)]
        for future in as_completed(futures):
            result, done = future.result(), done + 1
            if "error" in result:
                errors += 1
                print(f"  list failed: {result['error']}")
                if errors >= 20 and errors > done / 2:  # the stack is down, not one bad list
                    pool.shutdown(cancel_futures=True)
                    sys.exit("Stopping: most lists are failing. Check `docker compose ps`.")
            if result["version"] is not None:
                t = totals[result["version"]]
                t["lists"] += 1
                t["like"], t["skip"] = t["like"] + result["like"], t["skip"] + result["skip"]
                explored[0] += result["explored_likes"]
                explored[1] += result["explored"]
            if done % max(1, args.lists // 10) == 0:
                likes = sum(t["like"] for t in totals.values())
                rated = likes + sum(t["skip"] for t in totals.values())
                rate = f"{100 * likes / rated:.1f}%" if rated else "—"
                print(f"  {done:,}/{args.lists:,} lists, like rate so far {rate}")
    print(summary(totals, explored, errors, args.lists, time.monotonic() - start))


if __name__ == "__main__":
    main()
