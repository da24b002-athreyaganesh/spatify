"""Ingest stage: read the base catalogue and every new batch into one Ray dataset.

Every column is read as text, so a malformed value can't crash the read; the validate
stage does the type checks. Each row keeps the name of the file it came from (`source`),
so rejections and drift can be traced back to a batch.

Run:  python -m spatify.pipeline.ingest
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import ray.data as rd
from pyarrow import csv as pacsv

from spatify.pipeline.common import fresh_dir, init_ray, setup_logging, throughput, write_metrics
from spatify.schema import COLUMNS

log = logging.getLogger(__name__)


def input_files(base: Path, incoming: Path) -> list[Path]:
    """The base CSV, then every CSV in the incoming folder in name order."""
    return [base, *sorted(incoming.glob("*.csv"))]


def _tag_source(batch: pd.DataFrame) -> pd.DataFrame:
    """Replace Ray's full `path` column with just the file name."""
    batch["source"] = batch.pop("path").map(lambda p: Path(p).name)
    return batch


def read_tracks(paths: list[Path]) -> rd.Dataset:
    """Read track CSVs as text columns, with a `source` column naming each row's file."""
    convert = pacsv.ConvertOptions(
        column_types=dict.fromkeys(COLUMNS, pa.string()), strings_can_be_null=True
    )
    # Absolute paths: Ray's workers don't share the driver's working directory.
    files = [str(p.resolve()) for p in paths]
    ds = rd.read_csv(files, include_paths=True, convert_options=convert)
    return ds.map_batches(_tag_source, batch_format="pandas")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Read the base set and new batches.")
    parser.add_argument("--base", type=Path, default=Path("data/base/tracks.csv"))
    parser.add_argument("--incoming", type=Path, default=Path("data/incoming"))
    parser.add_argument("--out", type=Path, default=Path("data/ingested"))
    parser.add_argument("--metrics", type=Path, default=Path("metrics/ingest.json"))
    args = parser.parse_args(argv)
    setup_logging()

    paths = input_files(args.base, args.incoming)
    log.info("Ingesting %d files: %s", len(paths), ", ".join(p.name for p in paths))
    init_ray()

    start = time.perf_counter()
    ds = read_tracks(paths).materialize()
    ds.write_parquet(str(fresh_dir(args.out).resolve()))
    seconds = time.perf_counter() - start

    metrics = {"files": len(paths), **throughput(ds.count(), seconds)}
    write_metrics(args.metrics, metrics)
    log.info("Ingested %(rows)d rows in %(seconds).1f s (%(rows_per_s).0f rows/s)", metrics)


if __name__ == "__main__":
    main()
