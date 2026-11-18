# Data card: Spotify Tracks Dataset

| | |
|---|---|
| Source | maharshipandya, "Spotify Tracks Dataset" (Kaggle). Downloaded from the author's Hugging Face copy: https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset |
| File | `dataset.csv`, 20,118,244 bytes, SHA-256 `b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd` (downloaded 2026-10-23) |
| Licence | BSD, as stated on the dataset card |
| Tracked by | `data/raw/spotify_tracks.csv.dvc` (`dvc import-url`, which records the source URL) |

## Contents

- 114,000 rows and 21 columns: an unnamed row index, `track_id`, `artists`, `album_name`, `track_name`, `popularity`, `duration_ms`, `explicit`, 11 audio features (`danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`), `time_signature` and `track_genre`.
- 114 genres with exactly 1,000 rows each. The dataset card says 125.
- 89,741 unique track IDs; 16,299 tracks appear under more than one genre.
- 450 exact duplicate rows.
- One row has no `artists`, `album_name` or `track_name`.
- `tempo`, `duration_ms` and `time_signature` contain zeros, which are probably missing values rather than measurements. Zero tempo and time signature are kept with a warning, since they are mostly silent "sleep" tracks (D10).
- No release-year or date column, so the problem statement's "era" shift is simulated (the `drift` settings in `params.yaml`).

## Known biases and limits

- The genre balance is artificial: 1,000 tracks per genre, whatever the genre's real catalogue size or listening share.
- Genre labels overlap (16,299 tracks carry several), which affects any genre-based metric. Such tracks are merged into one row with every genre listed, and a recommendation counts as a genre match when it shares any genre with the seed (D11, D17).
- `popularity` reflects plays at collection time; the card notes that tracks played a lot now score higher than tracks played a lot in the past.
- These are audio features only: no listening history, so there are no real user preferences to evaluate against. Feedback comes from a simulation script (stated in the README and the report).
