"""Throughput benchmark: the Ray pipeline against plain pandas on N copies of the base set.

Both paths run the same logic (validate.check_rows, features.collapse_tracks,
features.add_engineered, standard scaling) from CSV files to Parquet, so the difference
is the engine. Ray gets one untimed warm-up run first, as a long-running worker would.
Results go to reports/benchmark.json and reports/benchmark.md.

Run:  python scripts/benchmark.py --copies 1 10      (one or more dataset sizes)
For Ray Dashboard screenshots, start a cluster first and point the script at it:
    ray start --head          then          RAY_ADDRESS=auto python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import ray

from spatify.pipeline import features, ingest, validate
from spatify.pipeline.common import init_ray, load_params


def make_copies(base: Path, n: int, folder: Path) -> list[Path]:
    """Write n identical copies of the base CSV, the "10 copies of the dataset" workload."""
    paths = [folder / f"copy_{i:02d}.csv" for i in range(n)]
    for path in paths:
        shutil.copy(base, path)
    return paths


def run_ray(paths: list[Path], out: Path, ranges: dict, columns: list[str]) -> float:
    start = time.perf_counter()
    ds = ingest.read_tracks(paths)
    _, valid, _ = validate.split(ds, ranges)
    tracks, _ = features.build(valid, columns)
    tracks.write_parquet(str(out.resolve()))
    return time.perf_counter() - start


def run_pandas(paths: list[Path], out: Path, ranges: dict, columns: list[str]) -> float:
    start = time.perf_counter()
    df = pd.concat(
        [pd.read_csv(p, dtype=str).assign(source=p.name) for p in paths], ignore_index=True
    )
    checked = validate.check_rows(df, ranges)
    valid = checked[checked["reject_reason"] == ""].drop(
        columns=["reject_reason", "reject_value", "warning"]
    )
    tracks = features.add_engineered(features.collapse_tracks(valid))
    for col in columns:
        tracks[f"z_{col}"] = (tracks[col] - tracks[col].mean()) / tracks[col].std(ddof=0)
    out.mkdir(parents=True)
    tracks.to_parquet(out / "tracks.parquet", index=False)
    return time.perf_counter() - start


def rate(rows: int, seconds: float) -> dict:
    """Seconds and rows per second for one run."""
    return {"seconds": round(seconds, 2), "rows_per_s": round(rows / seconds)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ray vs pandas throughput benchmark.")
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--base", type=Path, default=Path("data/base/tracks.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    ranges = load_params("validate")["ranges"]
    columns = load_params("features")["columns"]

    start = time.perf_counter()
    init_ray()
    startup = time.perf_counter() - start
    cpus = int(ray.cluster_resources().get("CPU", 0))
    base_rows = len(pd.read_csv(args.base))

    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = make_copies(args.base, max(args.copies), tmp)
        run_ray(paths[:1], tmp / "warmup", ranges, columns)
        for n in args.copies:
            rows = n * base_rows
            ray_seconds = run_ray(paths[:n], tmp / f"ray_{n}", ranges, columns)
            pandas_seconds = run_pandas(paths[:n], tmp / f"pandas_{n}", ranges, columns)
            runs.append(
                {
                    "copies": n,
                    "rows": rows,
                    "ray": rate(rows, ray_seconds),
                    "pandas": rate(rows, pandas_seconds),
                    "ray_speedup": round(pandas_seconds / ray_seconds, 2),
                }
            )
            print(json.dumps(runs[-1]))

    results = {
        "machine_cpus": os.cpu_count(),
        "ray_cpus": cpus,
        "ray_startup_seconds": round(startup, 2),
        "runs": runs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        f"| {r['copies']} | {r['rows']:,} | {r['ray']['rows_per_s']:,} | "
        f"{r['pandas']['rows_per_s']:,} | {r['ray_speedup']}x |"
        for r in runs
    ]
    (args.out / "benchmark.md").write_text(
        f"""# Pipeline throughput benchmark

Copies of `{args.base}` ({base_rows:,} rows each), CSV to Parquet through validate and
features. Machine: {results["machine_cpus"]} CPUs; Ray used {cpus}. Ray start-up
({results["ray_startup_seconds"]} s) and one warm-up run are not counted.

| Copies | Rows | Ray Data rows/s | pandas rows/s (one process) | Ray speed-up |
|---|---|---|---|---|
"""
        + "\n".join(lines)
        + "\n"
    )
    print(f"Wrote {args.out / 'benchmark.md'}")


if __name__ == "__main__":
    main()
