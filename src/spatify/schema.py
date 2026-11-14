"""Column definitions shared by the pipeline stages and, later, the services."""

# Every column a track file must have.
COLUMNS = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "track_genre",
]

# Descriptive text, kept for search and display; a missing value is only a warning.
METADATA = ["artists", "album_name", "track_name"]

# Numeric columns: each must be present, numeric, and inside its range in params.yaml.
NUMERIC = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

# Columns that must never be empty.
REQUIRED = ["track_id", "track_genre", "explicit", *NUMERIC]

# Numeric columns that hold whole numbers; validate reads everything as float first.
INTEGER = ["popularity", "duration_ms", "key", "mode", "time_signature"]
