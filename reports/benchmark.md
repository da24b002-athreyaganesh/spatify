# Pipeline throughput benchmark

Copies of `data/base/tracks.csv` (79,970 rows each), CSV to Parquet through validate and
features. Machine: 20 CPUs; Ray used 20. Ray start-up
(0.75 s) and one warm-up run are not counted.

| Copies | Rows | Ray Data rows/s | pandas rows/s (one process) | Ray speed-up |
|---|---|---|---|---|
| 1 | 79,970 | 14,702 | 56,821 | 0.26x |
| 10 | 799,700 | 82,083 | 104,254 | 0.79x |
| 20 | 1,599,400 | 134,076 | 114,928 | 1.17x |
