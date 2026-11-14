"""The gateway's view of the MLflow Model Registry: versions, aliases and the catalogue.

For the champion it also keeps the model's own catalogue (for search) and settings (for
the "why" labels, D33), both read from the files packaged inside the model version.
Uses mlflow-skinny (the client without the server), so the gateway image stays small.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from spatify.model.explain import explain
from spatify.model.recommender import QueryError, Recommender

log = logging.getLogger(__name__)

ALIASES = ("champion", "previous", "challenger")


class RegistryError(Exception):
    """A registry action that can't be done as asked (the gateway answers 409)."""


class Registry:
    def __init__(self, tracking_uri: str, name: str, client: MlflowClient | None = None) -> None:
        self.client = client or MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        self.tracking_uri = tracking_uri
        self.name = name
        self.champion_version: int | None = None
        self.catalogue: pd.DataFrame | None = None  # the champion's tracks, for search
        self._catalogue_version: int | None = None
        self._explainer: Recommender | None = None  # the champion's scaler and weights
        self._explainer_run: str | None = None
        self._run_versions: dict[str, int] = {}
        self._lock = threading.Lock()

    def aliases(self) -> dict[str, int]:
        """alias -> version; empty if the model isn't registered yet."""
        try:
            model = self.client.get_registered_model(self.name)
        except MlflowException:
            return {}
        return {alias: int(version) for alias, version in model.aliases.items()}

    def versions(self) -> list[dict]:
        """Every version, newest first, with its aliases and lineage tags."""
        by_version: dict[int, list[str]] = {}
        for alias, version in self.aliases().items():
            by_version.setdefault(version, []).append(alias)
        found = self.client.search_model_versions(f"name = '{self.name}'")
        rows = []
        for mv in sorted(found, key=lambda v: int(v.version), reverse=True):
            version = int(mv.version)
            self._run_versions[mv.run_id] = version
            rows.append(
                {
                    "version": version,
                    "aliases": sorted(by_version.get(version, [])),
                    "run_id": mv.run_id,
                    "created_at": datetime.fromtimestamp(mv.creation_timestamp / 1000, UTC),
                    "description": mv.description or None,
                    **{key: mv.tags.get(key) for key in ("variant", "dataset_version")},
                    **{key: mv.tags.get(key) for key in ("git_commit", "trigger")},
                }
            )
        return rows

    def version_of_run(self, run_id: str) -> int | None:
        """The registry version holding the model logged in this run (cached)."""
        if run_id not in self._run_versions:
            self.versions()  # refresh the cache
        return self._run_versions.get(run_id)

    def refresh(self) -> None:
        """Re-read the champion; download its catalogue if the champion changed."""
        champion = self.aliases().get("champion")
        self.champion_version = champion
        if champion is not None and champion != self._catalogue_version:
            self.load_catalogue(champion, *self._download_model_files(champion))

    def _download_model_files(self, version: int) -> tuple[pd.DataFrame, dict]:
        """catalogue.parquet and model_info.json from inside a model version (via MLflow)."""
        root = self.client.get_model_version_download_uri(self.name, str(version))
        with tempfile.TemporaryDirectory() as tmp:
            files = {}
            for name in ("catalogue.parquet", "model_info.json"):
                files[name] = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"{root}/artifacts/{name}",
                    dst_path=tmp,
                    tracking_uri=self.tracking_uri,
                )
            catalogue = pd.read_parquet(Path(files["catalogue.parquet"]))
            info = json.loads(Path(files["model_info.json"]).read_text())
        return catalogue, info

    def load_catalogue(
        self, version: int, catalogue: pd.DataFrame, info: dict | None = None
    ) -> None:
        """Make this version's tracks searchable and, given its settings, explainable."""
        explainer = Recommender.from_info(catalogue, info) if info else None
        searchable = catalogue.assign(  # lower-cased copies speed up search
            _name=catalogue["track_name"].fillna("").str.lower(),
            _artists=catalogue["artists"].fillna("").str.lower(),
        )
        with self._lock:
            self.catalogue, self._catalogue_version = searchable, version
            self._explainer, self._explainer_run = explainer, (info or {}).get("run_id")
        log.info("Loaded the catalogue of version %s (%d tracks)", version, len(catalogue))

    def why(self, query: dict, track_ids: list[str], run_id: str) -> list[list[str]]:
        """ "Why" labels per track (D33), or none when the answer came from a model other
        than the one loaded here (the champion just changed) or can't be explained."""
        with self._lock:
            explainer, loaded_run = self._explainer, self._explainer_run
        if explainer is None or loaded_run != run_id:
            return [[] for _ in track_ids]
        try:
            return explain(explainer, pd.DataFrame([query]), track_ids)
        except QueryError:
            return [[] for _ in track_ids]

    def search(self, query: str, limit: int) -> tuple[int, pd.DataFrame] | None:
        """Tracks whose title or artist contains the query; title-prefix matches first."""
        with self._lock:
            catalogue, version = self.catalogue, self._catalogue_version
        if catalogue is None or version is None:
            return None
        q = query.strip().lower()
        hits = catalogue[
            catalogue["_name"].str.contains(q, regex=False)
            | catalogue["_artists"].str.contains(q, regex=False)
        ]
        hits = hits.assign(_prefix=hits["_name"].str.startswith(q))
        hits = hits.sort_values(["_prefix", "popularity"], ascending=False, kind="stable")
        return version, hits.head(limit)

    def rollback(self, version: int | None = None) -> tuple[int, int]:
        """Make `version` (default: the `previous` alias) champion; the old champion becomes
        `previous`, so a second rollback undoes the first. Returns (champion, previous)."""
        aliases = self.aliases()
        current = aliases.get("champion")
        target = version if version is not None else aliases.get("previous")
        if current is None:
            raise RegistryError("there is no champion to roll back from")
        if target is None:
            raise RegistryError("there is no `previous` version; name one explicitly")
        if target == current:
            raise RegistryError(f"version {target} is already the champion")
        try:
            self.client.get_model_version(self.name, str(target))
        except MlflowException as err:
            raise RegistryError(f"version {target} doesn't exist") from err
        self.client.set_registered_model_alias(self.name, "champion", str(target))
        self.client.set_registered_model_alias(self.name, "previous", str(current))
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        self.client.set_model_version_tag(self.name, str(current), "rolled_back_at", stamp)
        log.warning("Rolled back: champion %s -> %s", current, target)
        self.champion_version = target
        return target, current
