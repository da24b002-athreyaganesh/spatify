"""Deploy a deliberately broken model, to show a bad deployment and its rollback (D39).

The problem statement's evaluation plan asks for "successful rollback after a forced bad
deployment". This registers a copy of the champion whose audio features are shuffled
between tracks, as a bad join would do, and makes it champion directly, skipping the
promotion gate that would have stopped it. The old champion becomes `previous`.

What follows: the model server switches within about 15 s; simulated listeners like
about 6 in 100 songs instead of about 4 in 10 (measured offline, D39); the like rate falls
on the console and feedback drift fires; an operator rolls back (console: Models ->
Roll back, or POST /models/rollback), and the old champion answers again.

Run with the stack up, from spatify/:
    MLFLOW_TRACKING_URI=http://localhost:5000 uv run python scripts/force_bad_model.py --yes
"""

from __future__ import annotations

import argparse
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from spatify.model.pyfunc import log_candidate
from spatify.model.recommender import Recommender
from spatify.model.tracking import lineage
from spatify.pipeline.common import load_params


def scrambled(rec: Recommender, seed: int) -> Recommender:
    """The same model with each track's audio features taken from another, random track."""
    catalogue = rec.catalogue.copy()
    shuffled = catalogue[rec.columns].sample(frac=1, random_state=seed).to_numpy()
    catalogue[rec.columns] = shuffled
    return Recommender(catalogue, rec.columns, rec.mean, rec.std, rec.weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a deliberately broken model champion.")
    parser.add_argument("--yes", action="store_true", help="really do it")
    parser.add_argument("--seed", type=int, default=0, help="the shuffle's seed")
    args = parser.parse_args()
    name = load_params("register")["model_name"]
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        sys.exit(f"No champion for model {name!r}: run the pipeline first.")
    if not args.yes:
        sys.exit(
            f"This makes a broken copy of version {champion.version} the champion, skipping "
            "the promotion gate. Listeners get poor recommendations until you roll back. "
            "Pass --yes to go ahead."
        )

    model = mlflow.pyfunc.load_model(f"models:/{name}@champion").unwrap_python_model()
    broken = scrambled(model.recommender, args.seed)
    tags = {
        **lineage(),
        "dataset_version": champion.tags.get("dataset_version", "unknown"),
        "trigger": "forced_bad",
        "variant": "forced-bad",
    }
    mlflow.set_experiment("spatify")
    with mlflow.start_run(run_name="forced-bad") as run:
        mlflow.set_tags({**tags, "stage": "train", "base_version": champion.version})
        mlflow.log_params({"base_version": champion.version, "shuffle_seed": args.seed})
        info = {"run_id": run.info.run_id, "variant": "forced-bad"}
        model_info = log_candidate(broken, info | {"dataset_version": tags["dataset_version"]},
                                   model.default_k)  # fmt: skip
    version = str(mlflow.register_model(model_info.model_uri, name).version)
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    client.update_model_version(
        name,
        version,
        description=(
            f"Deliberately broken copy of version {champion.version} (features shuffled "
            "between tracks), made champion without the promotion gate for the rollback demo"
        ),
    )
    client.set_registered_model_alias(name, "previous", champion.version)
    client.set_registered_model_alias(name, "champion", version)
    print(
        f"Version {version} (broken) is now champion; version {champion.version} is previous.\n"
        "The model server switches within about 15 s. Run the simulator to see the like rate "
        "fall, then roll back:\n"
        "  console: Operator console -> Models -> Roll back (to the previous version)\n"
        '  or: curl -X POST http://localhost:8000/models/rollback -H "X-API-Key: $SPATIFY_API_KEY"'
    )


if __name__ == "__main__":
    main()
