"""The cold-start recommender: cosine nearest neighbours over scaled, weighted audio features.

Plain NumPy and pandas, so the served model needs neither Ray nor the training code.

A query is a seed track, slider values, or both:
- a seed track: its own features. The seed, and other releases of the same song (same
  artists and title under another track_id), are left out of the results;
- sliders only: the slider values, with every feature left unset at the catalogue mean.
  That is z = 0 after scaling, so an unset feature adds no direction to the query and
  doesn't pull the results anywhere (decision D19);
- both: the seed's features, with the given sliders overriding them
  ("like this track, but more energetic").
A query with neither is rejected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features a user can set directly, in their raw units. The key is not one of them: as a
# point on the circle of fifths it only makes sense taken from a seed track.
SLIDERS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "mode",
]

# Track details returned with every recommendation.
DETAILS = ["track_id", "track_name", "artists", "album_name", "genres", "popularity"]

# Queries scored per matrix product; caps memory at CHUNK x catalogue-size floats.
CHUNK = 256


class QueryError(ValueError):
    """A query the model can't answer; the server reports it as a bad request (400)."""


class UnknownTrackError(QueryError):
    """A seed track_id that isn't in the model's catalogue."""


class EmptyQueryError(QueryError):
    """A query row with no seed track and no slider values."""


def fisher_weights(z: np.ndarray, genres: pd.Series) -> np.ndarray:
    """Per-feature weights from how well each feature separates genres (Fisher's ratio).

    `z` holds standardised features, one row per track; `genres` the ";"-separated genres
    of each track, and a track in several genres counts once in each. The ratio is the
    variance of the genre means over the mean within-genre variance. Each feature is
    multiplied by sqrt(ratio), so its share of the squared distance follows its ratio.
    The weights are scaled to a mean square of 1, the same overall scale as uniform weights.
    """
    pairs = genres.reset_index(drop=True).str.split(";").explode()
    rows = z[pairs.index.to_numpy()]
    genre_means = pd.DataFrame(rows).groupby(pairs.to_numpy()).transform("mean").to_numpy()
    within = ((rows - genre_means) ** 2).mean(axis=0)
    between = ((genre_means - rows.mean(axis=0)) ** 2).mean(axis=0)
    weights = np.sqrt(between / np.maximum(within, 1e-12))
    scale = np.sqrt(np.mean(weights**2))
    return weights / scale if scale > 0 else np.ones(z.shape[1])


def _unit_rows(x: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product is a cosine similarity."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Recommender:
    """Ranks catalogue tracks by cosine similarity to a query, in the weighted z-score space.

    `catalogue` holds DETAILS plus the raw (unscaled) values of `columns`. `mean` and `std`
    are the scaler from the features stage, so a query is scaled exactly as the catalogue
    was; `weights` multiply the scaled features.
    """

    def __init__(
        self,
        catalogue: pd.DataFrame,
        columns: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        self.catalogue = catalogue.reset_index(drop=True)
        self.columns = list(columns)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.where(np.asarray(std, dtype=np.float64) > 0, std, 1.0)
        self.weights = np.asarray(weights, dtype=np.float64)

        self.raw = self.catalogue[self.columns].to_numpy(dtype=np.float64)
        self.matrix = _unit_rows(self.transform(self.raw)).astype(np.float32)
        self.positions = pd.Index(self.catalogue["track_id"].astype(str))
        # One code per song (artists + title); a track with no name counts as its own song.
        song = self.catalogue["artists"] + "\x1f" + self.catalogue["track_name"]
        song = song.fillna("id:" + self.catalogue["track_id"].astype(str))
        self.song = pd.factorize(song)[0]

    @classmethod
    def from_info(cls, catalogue: pd.DataFrame, info: dict) -> Recommender:
        """Rebuild a recommender from its catalogue and the settings saved by `info()`."""
        cols = info["columns"]
        return cls(
            catalogue,
            cols,
            np.array([info["mean"][c] for c in cols]),
            np.array([info["std"][c] for c in cols]),
            np.array([info["weights"][c] for c in cols]),
        )

    def info(self) -> dict:
        """The settings needed to rebuild this recommender (saved with the model)."""
        return {
            "columns": self.columns,
            "mean": dict(zip(self.columns, self.mean.tolist(), strict=True)),
            "std": dict(zip(self.columns, self.std.tolist(), strict=True)),
            "weights": dict(zip(self.columns, self.weights.tolist(), strict=True)),
        }

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Raw feature values -> weighted z-scores."""
        return (raw - self.mean) / self.std * self.weights

    def query_features(self, queries: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Raw feature values for each query row, and its seed's catalogue position (-1: none)."""
        n = len(queries)
        raw = np.tile(self.mean, (n, 1))  # unset features: the mean, z = 0
        seeds = np.full(n, -1)
        answered = np.zeros(n, dtype=bool)
        if "track_id" in queries:
            ids = queries["track_id"].astype("string").str.strip()
            given = (ids.notna() & (ids != "")).fillna(False).to_numpy(dtype=bool)
            found = self.positions.get_indexer(ids[given].astype(str))
            if (found < 0).any():
                unknown = ids[given][found < 0].tolist()
                raise UnknownTrackError(f"Unknown seed track_id: {', '.join(unknown[:5])}")
            seeds[given] = found
            raw[given] = self.raw[found]
            answered |= given
        for col in SLIDERS:
            if col in queries:
                values = pd.to_numeric(queries[col], errors="coerce").to_numpy(dtype=np.float64)
                is_set = ~np.isnan(values)
                if col in self.columns:  # a feature this model doesn't use is ignored
                    raw[is_set, self.columns.index(col)] = values[is_set]
                    answered |= is_set
        if not answered.all():
            rows = np.flatnonzero(~answered).tolist()
            raise EmptyQueryError(
                f"Query rows {rows[:5]} have no track_id and no slider value this model uses"
            )
        return raw, seeds

    def recommend(self, queries: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """Top-k tracks per query row: `query` (row number), `rank`, DETAILS and `score`."""
        raw, seeds = self.query_features(queries)
        vectors = _unit_rows(self.transform(raw)).astype(np.float32)
        k = max(1, min(int(k), len(self.catalogue)))
        query_rows, ranks, picked, scores = [], [], [], []
        for start in range(0, len(vectors), CHUNK):
            similarity = vectors[start : start + CHUNK] @ self.matrix.T
            for offset, row in enumerate(similarity):
                top = self._top(row, k, seeds[start + offset])
                query_rows += [start + offset] * len(top)
                ranks += range(1, len(top) + 1)
                picked += top
                scores += row[top].tolist()
        result = self.catalogue.loc[picked, DETAILS].reset_index(drop=True)
        result.insert(0, "query", np.array(query_rows, dtype=np.int64))
        result.insert(1, "rank", np.array(ranks, dtype=np.int64))
        result["score"] = np.round(np.array(scores, dtype=np.float64), 4)
        return result

    def _top(self, similarity: np.ndarray, k: int, seed: int) -> list[int]:
        """The k most similar tracks, skipping the seed and repeats of a song already picked."""
        n = len(similarity)
        fetch = min(n, 4 * k + 8)  # extra candidates to cover skipped repeats
        while True:
            candidates = np.argpartition(-similarity, fetch - 1)[:fetch]
            candidates = candidates[np.argsort(-similarity[candidates], kind="stable")]
            seen = {self.song[seed]} if seed >= 0 else set()
            top: list[int] = []
            for i in candidates:
                if i == seed or self.song[i] in seen:
                    continue
                seen.add(self.song[i])
                top.append(int(i))
                if len(top) == k:
                    return top
            if fetch == n:
                return top
            fetch = min(n, fetch * 4)
