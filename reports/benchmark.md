# Pipeline throughput benchmark

Copies of `data/base/tracks.csv` (79,970 rows each), CSV to Parquet through validate and
features. Machine: 20 CPUs; Ray used 4. Ray start-up
(0.38 s) and one warm-up run are not counted.

| Copies | Rows | Ray Data rows/s | pandas rows/s (one process) | Ray speed-up |
|---|---|---|---|---|
| 1 | 79,970 | 18,436 | 53,287 | 0.35x |
| 10 | 799,700 | 66,103 | 99,732 | 0.66x |
