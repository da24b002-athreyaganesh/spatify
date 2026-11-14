"""Validate stage: check every row and split the data into valid and rejected rows.

Row checks; a row is rejected with the first reason that applies:
- a required column is empty             -> null:<column>
- a numeric column holds text            -> type:<column>
- a value is outside its range           -> range:<column>  (ranges in params.yaml)
Soft warnings; the row is kept and counted:
- no artist, album or title              -> missing:<column>
- zero tempo or time signature           -> zero:<column>  (silent tracks, mostly "sleep")
Hard failures; the stage exits with an error, so DVC stops the pipeline and the serving
model is left alone:
- a column is missing from the input
- any source file has more than `max_reject_rate` of its rows rejected. The rate is per
  file, so a bad 2,000-row batch can't hide inside the 80,000-row base set.

Run:  python -m spatify.pipeline.validate
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray.data as rd

from spatify.pipeline.common import (
    fresh_dir,
    init_ray,
    load_params,
    setup_logging,
    throughput,
    write_metrics,
)
from spatify.schema import COLUMNS, METADATA, NUMERIC, REQUIRED

log = logging.getLogger(__name__)


def check_rows(batch: pd.DataFrame, ranges: dict[str, list[float]]) -> pd.DataFrame:
    """Type the columns and add `reject_reason`, `reject_value` and `warning` ("" if none)."""
    out = batch.copy()
    conditions, reasons, values = [], [], []

    def check(condition: pd.Series, reason: str, column: str) -> None:
        conditions.append(condition.to_numpy(dtype=bool))
        reasons.append(reason)
        values.append(batch[column].astype("string").fillna("").to_numpy(dtype=object))

    for col in REQUIRED:
        check(batch[col].isna(), f"null:{col}", col)
    for col in NUMERIC:
        typed = pd.to_numeric(batch[col], errors="coerce").astype("float64")
        low, high = ranges[col]
        check(batch[col].notna() & typed.isna(), f"type:{col}", col)
        check(typed.notna() & ~typed.between(low, high), f"range:{col}", col)
        out[col] = typed
    explicit = batch["explicit"].astype("string").str.lower().map({"true": True, "false": False})
    check(batch["explicit"].notna() & explicit.isna(), "type:explicit", "explicit")
    out["explicit"] = explicit.astype("boolean")

    out["reject_reason"] = np.select(conditions, reasons, default="")
    out["reject_value"] = np.select(conditions, values, default="")
    zero_cols = ["tempo", "time_signature"]
    warnings = [out[c].isna() for c in METADATA] + [out[c] == 0 for c in zero_cols]
    labels = [f"missing:{c}" for c in METADATA] + [f"zero:{c}" for c in zero_cols]
    out["warning"] = np.select([w.to_numpy(dtype=bool) for w in warnings], labels, default="")
    return out


def _valid_rows(batch: pd.DataFrame) -> pd.DataFrame:
    keep = batch["reject_reason"] == ""
    return batch[keep].drop(columns=["reject_reason", "reject_value", "warning"])


def _rejected_rows(batch: pd.DataFrame) -> pd.DataFrame:
    return batch[batch["reject_reason"] != ""]


def split(ds: rd.Dataset, ranges: dict) -> tuple[rd.Dataset, rd.Dataset, rd.Dataset]:
    """Check every row in parallel; return the checked, valid and rejected datasets."""
    checked = ds.map_batches(
        check_rows, fn_kwargs={"ranges": ranges}, batch_format="pandas"
    ).materialize()
    valid = checked.map_batches(_valid_rows, batch_format="pandas")
    rejected = checked.map_batches(_rejected_rows, batch_format="pandas")
    return checked, valid, rejected


def summarise(checked: rd.Dataset) -> pd.DataFrame:
    """Row counts per source file, reject reason and warning."""
    counts = checked.groupby(["source", "reject_reason", "warning"]).count().to_pandas()
    return counts.rename(columns={"count()": "rows"})


def report(counts: pd.DataFrame, max_reject_rate: float) -> dict:
    """Build the validation report (the DVC metrics) from the per-group row counts."""
    is_rejected = counts["reject_reason"] != ""
    per_source = counts.groupby("source")["rows"].sum()
    rejected = (
        counts[is_rejected].groupby("source")["rows"].sum().reindex(per_source.index, fill_value=0)
    )
    rates = rejected / per_source
    kept_warnings = counts[~is_rejected & (counts["warning"] != "")]
    total = int(per_source.sum())
    return {
        "rows": total,
        "valid": total - int(rejected.sum()),
        "rejected": int(rejected.sum()),
        "reject_rate": round(float(rejected.sum()) / total, 4),
        "max_reject_rate": max_reject_rate,
        "failed_sources": sorted(rates[rates > max_reject_rate].index),
        "rejected_by_source": {s: int(n) for s, n in rejected.items()},
        "reject_rate_by_source": {s: round(float(r), 4) for s, r in rates.items()},
        "rejected_by_reason": {
            r: int(n) for r, n in counts[is_rejected].groupby("reject_reason")["rows"].sum().items()
        },
        "warnings": {w: int(n) for w, n in kept_warnings.groupby("warning")["rows"].sum().items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate the ingested rows.")
    parser.add_argument("--data", type=Path, default=Path("data/ingested"))
    parser.add_argument("--out", type=Path, default=Path("data/validated"))
    parser.add_argument("--metrics", type=Path, default=Path("metrics/validate.json"))
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_params("validate")
    init_ray()

    start = time.perf_counter()
    ds = rd.read_parquet(str(args.data.resolve()))  # absolute: Ray workers have their own cwd
    missing = sorted(set(COLUMNS) - set(ds.columns()))
    if missing:
        log.error("Input is missing columns %s; stopping the pipeline", missing)
        raise SystemExit(1)

    checked, valid, rejected = split(ds, cfg["ranges"])
    fresh_dir(args.out)
    valid.write_parquet(str((args.out / "tracks").resolve()))
    rejected.to_pandas().to_csv(args.out / "rejected.csv", index=False)
    summary = report(summarise(checked), cfg["max_reject_rate"])
    seconds = time.perf_counter() - start

    metrics = {**summary, **throughput(summary["rows"], seconds)}
    write_metrics(args.metrics, metrics)
    log.info(
        "%d rows: %d valid, %d rejected (%.2f%%) in %.1f s (%.0f rows/s)",
        metrics["rows"],
        metrics["valid"],
        metrics["rejected"],
        100 * metrics["reject_rate"],
        metrics["seconds"],
        metrics["rows_per_s"],
    )
    if metrics["failed_sources"]:
        log.error(
            "Reject rate above %.0f%% in %s; stopping the pipeline. "
            "See %s for the rejected rows and their reasons.",
            100 * cfg["max_reject_rate"],
            ", ".join(metrics["failed_sources"]),
            args.out / "rejected.csv",
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
