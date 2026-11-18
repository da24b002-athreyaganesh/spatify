"""Load test: /recommend latency at 1, 10 and 40 concurrent clients (test plan TC-PERF, D43).

The course's closed-loop pattern (the A5 / KServe lab load_test.py): each client thread
sends a request, waits for the answer and sends the next, for `--duration` seconds per
level. Requests are realistic listener queries (seed songs and sliders, as the simulator
makes them) and send no feedback, so the test can't trigger a retrain.

Run with the stack up, from spatify/ (about 2 minutes):
    uv run python scripts/load_test.py 2>&1 | tee evidence/phase7_load_test.log

Writes reports/load_test/load_test.{json,md,png}: per level the requests, throughput,
p50/p95/p99 latency, errors and the model versions that answered, Little's law
(R = N / X), and a plot of the percentiles against concurrency with the 300 ms target.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from spatify.model.data import load_tracks
from spatify.model.plots import BLUE, GREY, ORANGE, plt
from spatify.simulation import new_query

TARGET_MS = 300  # the problem statement's "low hundreds of milliseconds" at p95
PURPLE, VERMILLION = "#CC79A7", "#D55E00"  # Okabe-Ito, as in the dashboards


def percentile(sorted_ms: list[float], p: float) -> float:
    """The A5 script's nearest-rank percentile of an ascending list."""
    return sorted_ms[min(len(sorted_ms) - 1, int(p * len(sorted_ms)))]


def run_level(url: str, clients: int, seconds: float, ids: np.ndarray, seed: int) -> dict:
    """`clients` threads in a closed loop for `seconds`; one result row."""
    end = time.monotonic() + seconds

    def client(number: int) -> list[tuple[float, bool, object]]:
        rng = np.random.default_rng([seed, clients, number])
        out = []
        with httpx.Client(base_url=url, timeout=30) as api:
            while time.monotonic() < end:
                body = {**new_query(rng, ids, 0.3), "k": 10}
                start = time.perf_counter()
                try:
                    answer = api.post("/recommend", json=body)
                    ok = answer.status_code == 200
                    version = answer.json().get("model_version") if ok else None
                except httpx.HTTPError:
                    ok, version = False, None
                out.append((1000 * (time.perf_counter() - start), ok, version))
        return out

    with cf.ThreadPoolExecutor(clients) as pool:
        results = [r for f in [pool.submit(client, i) for i in range(clients)] for r in f.result()]
    ms = sorted(r[0] for r in results)
    n = len(ms)
    rps = n / seconds
    row = {
        "concurrency": clients,
        "requests": n,
        "rps": round(rps, 1),
        "p50_ms": round(percentile(ms, 0.50), 1),
        "p95_ms": round(percentile(ms, 0.95), 1),
        "p99_ms": round(percentile(ms, 0.99), 1),
        "errors": sum(not r[1] for r in results),
        "versions": {str(k): v for k, v in Counter(r[2] for r in results if r[1]).items()},
        "little_r_ms": round(1000 * clients / rps, 1) if rps else None,
    }
    row["meets_target"] = row["p95_ms"] <= TARGET_MS and row["errors"] == 0
    return row


def table(rows: list[dict]) -> str:
    lines = [
        "| clients | requests | req/s | p50 ms | p95 ms | p99 ms | errors | Little's R = N/X (ms) "
        f"| p95 <= {TARGET_MS} ms, no errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['concurrency']} | {r['requests']:,} | {r['rps']} | {r['p50_ms']} | "
            f"{r['p95_ms']} | {r['p99_ms']} | {r['errors']} | {r['little_r_ms']} | "
            f"{'yes' if r['meets_target'] else '**no**'} |"
        )
    return "\n".join(lines)


def figure(rows: list[dict], seconds: float):
    """p50/p95/p99 against concurrency, the target line, and a caption (A4's commandments)."""
    clients = [r["concurrency"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, colour, style, marker in (
        ("p50_ms", "p50 (median)", BLUE, "-", "o"),
        ("p95_ms", "p95", ORANGE, "--", "s"),
        ("p99_ms", "p99", PURPLE, ":", "^"),
    ):
        ax.plot(clients, [r[key] for r in rows], color=colour, linestyle=style, marker=marker,
                linewidth=2, label=label)  # fmt: skip
    ax.axhline(TARGET_MS, color=VERMILLION, linestyle="-.", linewidth=1)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], color=VERMILLION, linestyle="-.", label=f"target: p95 {TARGET_MS} ms")
    )
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(clients, [str(c) for c in clients])
    ax.minorticks_off()
    ax.set_xlabel("concurrent clients (closed loop, log scale)")
    ax.set_ylabel("time to answer POST /recommend (ms, linear)")
    ax.set_ylim(bottom=0)
    ax.set_title("Spatify /recommend latency against concurrency")
    fig.text(
        0.01, 0.01,
        f"{seconds:.0f} s per level; each client sends its next request as soon as the last "
        "one is answered. At or below the dash-dotted line meets the target.",
        fontsize=8, color=GREY, ha="left", va="bottom", wrap=True,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop load test of /recommend.")
    parser.add_argument("--url", default="http://localhost:8000", help="the gateway")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 40])
    parser.add_argument("--duration", type=float, default=30, help="seconds per level")
    parser.add_argument("--warmup", type=float, default=5, help="seconds, not counted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=Path("data/features/tracks"))
    parser.add_argument("--out", type=Path, default=Path("reports/load_test"))
    args = parser.parse_args()

    try:
        health = httpx.get(f"{args.url}/health", timeout=10)
    except httpx.HTTPError as err:
        sys.exit(f"Can't reach the gateway at {args.url} ({err}); is the stack up?")
    if health.status_code != 200:
        sys.exit(f"The gateway isn't ready: {health.text}")
    ids = load_tracks(args.data)["track_id"].astype(str).to_numpy()

    print(f"Warm-up: {args.warmup:.0f} s at 1 client (not counted)")
    run_level(args.url, 1, args.warmup, ids, args.seed)
    rows = []
    for clients in args.concurrency:
        print(f"{clients} clients for {args.duration:.0f} s ...", flush=True)
        rows.append(run_level(args.url, clients, args.duration, ids, args.seed))
        r = rows[-1]
        print(f"  {r['requests']:,} requests, {r['rps']} req/s, p50 {r['p50_ms']} ms, "
              f"p95 {r['p95_ms']} ms, p99 {r['p99_ms']} ms, {r['errors']} errors, "
              f"versions {r['versions']}")  # fmt: skip

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "load_test.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out / "load_test.md").write_text(table(rows) + "\n")
    fig = figure(rows, args.duration)
    fig.savefig(args.out / "load_test.png", dpi=120)
    plt.close(fig)
    print("\n" + pd.DataFrame(rows).drop(columns="versions").to_string(index=False))
    print("\n" + table(rows))
    print(f"\nWrote {args.out}/load_test.json, load_test.md and load_test.png")


if __name__ == "__main__":
    main()
