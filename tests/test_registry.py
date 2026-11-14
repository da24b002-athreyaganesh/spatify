"""Tests for the packaged model and the promotion rule, against a throwaway MLflow registry."""

from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from test_recommender import recommender

from spatify.model.pyfunc import log_candidate
from spatify.model.register import alias_version, promote, should_promote

NAME = "spatify-test"


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """A SQLite tracking store and registry in a temporary folder, with two model versions."""
    tmp = tmp_path_factory.mktemp("mlflow")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)  # artifacts go to ./mlruns
        mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp / 'mlflow.db'}")
        mlflow.set_experiment("test")
        uris = []
        for _ in range(2):
            with mlflow.start_run() as run:
                info = {"run_id": run.info.run_id, "variant": "test", "dataset_version": "x"}
                uris.append((run.info.run_id, log_candidate(recommender(), info, k=3).model_uri))
        for _, uri in uris:
            mlflow.register_model(uri, NAME)
        yield MlflowClient(), uris


def test_promotion_rule():
    assert should_promote(0.12, None, 0.005)  # no champion yet
    assert should_promote(0.130, 0.125, 0.005)  # wins by exactly the minimum
    assert not should_promote(0.129, 0.125, 0.005)  # wins, but by too little
    assert not should_promote(0.10, 0.12, 0.005)  # loses


def test_packaged_model_answers_through_pyfunc(registry):
    _, uris = registry
    run_id, uri = uris[0]
    model = mlflow.pyfunc.load_model(uri)
    recs = model.predict(pd.DataFrame({"track_id": ["a"]}))
    assert len(recs) == 3  # the default k saved with the model
    assert (recs["model_run_id"] == run_id).all()  # every answer names its model
    assert len(model.predict(pd.DataFrame({"energy": [0.1]}), params={"k": 2})) == 2


def test_packaged_model_rejects_unknown_tracks(registry):
    _, uris = registry
    model = mlflow.pyfunc.load_model(uris[0][1])
    with pytest.raises(MlflowException, match="Unknown seed track_id"):
        model.predict(pd.DataFrame({"track_id": ["nope"]}))


def test_model_ships_only_its_own_requirements(registry):
    _, uris = registry
    folder = Path(mlflow.artifacts.download_artifacts(uris[0][1]))
    packages = {line.split("==")[0] for line in (folder / "requirements.txt").read_text().split()}
    assert packages == {"mlflow", "numpy", "pandas", "pyarrow"}
    assert not (folder / "uv.lock").exists()


def test_promote_moves_champion_and_keeps_previous(registry):
    client, _ = registry
    promote(client, NAME, "1", champion=None)  # first model: nothing to keep
    assert alias_version(client, NAME, "champion") == "1"
    assert alias_version(client, NAME, "previous") is None

    client.set_registered_model_alias(NAME, "challenger", "2")
    promote(client, NAME, "2", champion="1")
    assert alias_version(client, NAME, "champion") == "2"
    assert alias_version(client, NAME, "previous") == "1"  # the rollback target
    assert alias_version(client, NAME, "challenger") is None
