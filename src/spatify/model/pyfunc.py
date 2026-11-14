"""The recommender packaged as an MLflow pyfunc model: what the model server loads.

The model folder carries everything needed to answer a query (L10's custom-pyfunc pattern):
- `catalogue.parquet`: the tracks that can be recommended, with their raw feature values;
- `model_info.json`: feature columns, the features stage's scaler, the weights, the default
  K and the training run's ID;
- `code/spatify/`: this package, so the server can import the class without installing it.

Every answer includes `model_run_id`, the training run of the model that produced it, so
each feedback event can be traced to a model (the registry maps a run to its version).
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema

import spatify
from spatify.model.recommender import SLIDERS, QueryError, Recommender
from spatify.model.tracking import model_requirements

PACKAGE_DIR = Path(spatify.__file__).parent


class SpatifyModel(mlflow.pyfunc.PythonModel):
    """Answers a seed-track and/or slider query with the top-K tracks."""

    def load_context(self, context) -> None:
        info = json.loads(Path(context.artifacts["model_info"]).read_text())
        catalogue = pd.read_parquet(context.artifacts["catalogue"])
        self.recommender = Recommender.from_info(catalogue, info)
        self.run_id = info["run_id"]
        self.default_k = info["k"]

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None):
        k = (params or {}).get("k") or self.default_k
        try:
            result = self.recommender.recommend(pd.DataFrame(model_input), int(k))
        except QueryError as err:  # unknown seed track, or nothing to search with
            # INVALID_PARAMETER_VALUE makes the scoring server answer 400, not 500.
            raise MlflowException(str(err), error_code=INVALID_PARAMETER_VALUE) from err
        result["model_run_id"] = self.run_id
        return result


def signature(example_output: pd.DataFrame, k: int) -> ModelSignature:
    """The serving contract: every input column is optional; K is an inference parameter.

    A request sends a `track_id`, any of the slider features, or both. Sliders are doubles,
    so JSON integers and floats are both accepted.
    """
    inputs = Schema(
        [ColSpec("string", "track_id", required=False)]
        + [ColSpec("double", name, required=False) for name in SLIDERS]
    )
    params = ParamSchema([ParamSpec("k", "long", k)])
    with warnings.catch_warnings():
        # MLflow warns that integer columns can't hold missing values; that matters for
        # inputs, and the integer columns here (query, rank, popularity) are outputs.
        warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
        outputs = mlflow.models.infer_signature(model_output=example_output).outputs
    return ModelSignature(inputs=inputs, outputs=outputs, params=params)


@contextmanager
def _env(name: str, value: str) -> Iterator[None]:
    """Set an environment variable for the duration of a block."""
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def log_candidate(recommender: Recommender, info: dict, k: int) -> ModelInfo:
    """Log the recommender as a pyfunc model in the active run, with signature and example."""
    example = pd.DataFrame(
        {"track_id": [str(recommender.catalogue.loc[0, "track_id"])], "energy": [0.8]}
    )
    example_output = recommender.recommend(example, k).assign(model_run_id=info["run_id"])
    # MLFLOW_UV_AUTO_DETECT off: MLflow 3.17 would otherwise copy the project's uv.lock
    # into the model, a second environment spec (the whole project, without mlflow)
    # next to the four pinned requirements the model actually needs.
    with tempfile.TemporaryDirectory() as tmp, _env("MLFLOW_UV_AUTO_DETECT", "false"):
        catalogue = Path(tmp) / "catalogue.parquet"
        recommender.catalogue.to_parquet(catalogue, index=False)
        model_info = Path(tmp) / "model_info.json"
        model_info.write_text(json.dumps({**recommender.info(), **info, "k": k}, indent=2))
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=SpatifyModel(),
            artifacts={"catalogue": str(catalogue), "model_info": str(model_info)},
            code_paths=[str(PACKAGE_DIR)],
            signature=signature(example_output, k),
            input_example=example,
            pip_requirements=model_requirements(),
        )
