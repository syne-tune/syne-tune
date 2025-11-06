"""
Convert evaluations from
 A Quantile-based Approach for Hyperparameter Transfer Learning
 David Salinas Huibin Shen Valerio Perrone
 http://proceedings.mlr.press/v119/salinas20a/salinas20a.pdf
"""
import pandas as pd
import numpy as np
from pathlib import Path
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path
import syne_tune.config_space as sp
from syne_tune.util import dump_json_with_numpy

from syne_tune.blackbox_repository.serialize import (
    serialize_configspace,
    serialize_metadata,
)
from syne_tune.config_space import (
    config_space_to_json_dict, randint,
)


def download(blackbox: str):
    import urllib

    root = "https://github.com/geoalgo/A-Quantile-based-Approach-for-Hyperparameter-Transfer-Learning/blob/master/src/blackbox/offline_evaluations/"
    urllib.request.urlretrieve(
        root + f"{blackbox}.csv.zip?raw=true", repository_path / f"{blackbox}.csv.zip"
    )

def serialize(
    bb_dict: dict[str, BlackboxTabular], path: str, metadata: dict | None = None
):
    # check all blackboxes share the objectives
    bb_first = next(iter(bb_dict.values()))
    for bb in bb_dict.values():
        assert bb.objectives_names == bb_first.objectives_names

    path = Path(path)
    path.mkdir(exist_ok=True)

    serialize_configspace(
        path=path,
        configuration_space=bb_first.configuration_space,
    )

    for task, bb in bb_dict.items():
        bb.hyperparameters.to_parquet(
            path / f"{task}-hyperparameters.parquet",
            index=False,
            compression="gzip",
            engine="fastparquet",
        )

        dump_json_with_numpy(
            config_space_to_json_dict(bb_dict[task].fidelity_space),
            filename=path / f"{task}-fidelity_space.json",
        )

        with open(path / f"{task}-objectives_evaluations.npy", "wb") as f:
            np.save(
                f,
                bb_dict[task].objectives_evaluations.astype(np.float32),
                allow_pickle=False,
            )

        with open(path / f"{task}-fidelity_values.npy", "wb") as f:
            np.save(f, bb_dict[task].fidelity_values, allow_pickle=False)

    metadata = metadata.copy() if metadata else {}
    metadata.update(
        {
            "objectives_names": bb_first.objectives_names,
            "task_names": list(bb_dict.keys()),
        }
    )
    serialize_metadata(
        path=path,
        metadata=metadata,
    )


def serialize_deepar():
    blackbox = "DeepAR"
    download(blackbox=blackbox)
    df = pd.read_csv(repository_path / f"{blackbox}.csv.zip")

    df["hp_num_layers"] = df.hp_num_layers.apply(np.exp)
    df["hp_num_cells"] = df.hp_num_cells.apply(np.exp)
    df["hp_dropout_rate"] = df.hp_dropout_rate_log.apply(np.exp)
    df["hp_learning_rate"] = df.hp_learning_rate_log.apply(np.exp)
    df["hp_num_batches_per_epoch"] = df.hp_num_batches_per_epoch_log.apply(np.exp)
    df["hp_context_length_ratio"] = df.hp_context_length_ratio_log.apply(np.exp)

    df = df[[col for col in df.columns if not col.endswith("_log")]]
    configuration_space = {
        "hp_num_layers": sp.randint(lower=2, upper=4),
        "hp_num_cells": sp.randint(lower=30, upper=120),
        "hp_dropout_rate": sp.uniform(lower=0.01, upper=0.51),
        "hp_learning_rate": sp.loguniform(lower=1e-4, upper=1e-2),
        "hp_num_batches_per_epoch": sp.lograndint(lower=10, upper=10000),
        "hp_context_length_ratio": sp.loguniform(lower=0.05, upper=4),
    }
    df["resource"] = 1  # Dummy fidelity

    bb_dict = {}
    for task in df.task.unique():
        df_task = df.loc[df.task == task]
        hypers = df_task[[col for col in df_task.columns if col.startswith("hp_")]]
        evals = df_task[[col for col in df_task.columns if col.startswith("metric_")]].values[:, None, None, :]
        bb_dict[task] = BlackboxTabular(
            hyperparameters=hypers,
            objectives_evaluations=evals,
            configuration_space=configuration_space,
            fidelity_space={'resource': randint(1, 1)},
            objectives_names=[
                col for col in df.columns if col.startswith("metric_")
            ],
        )
    serialize(
            bb_dict=bb_dict,
            path=repository_path / "icml-deepar",
        )


def serialize_xgboost():
    """
    'hp_log2_min_child_weight', 'hp_subsample', 'hp_colsample_bytree',
    'hp_log2_gamma', 'hp_log2_lambda', 'hp_eta', 'hp_max_depth_index',
    'hp_log2_alpha', 'metric_error', 'blackbox', 'task'
    """
    blackbox = "XGBoost"
    download(blackbox=blackbox)
    df = pd.read_csv(repository_path / f"{blackbox}.csv.zip")

    for hp in [
        "hp_log2_min_child_weight",
        "hp_log2_gamma",
        "hp_log2_lambda",
        "hp_log2_alpha",
    ]:
        df[hp.replace("_log2", "")] = df[hp].apply(np.exp2)

    df = df[[col for col in df.columns if not "_log2" in col]]

    configuration_space = {
        "hp_subsample": sp.uniform(lower=0.5, upper=1.0),
        "hp_colsample_bytree": sp.uniform(
            lower=0.3,
            upper=1.0,
        ),
        "hp_eta": sp.uniform(lower=0.0, upper=1.0),
        "hp_max_depth_index": sp.uniform(
            lower=0.0,
            upper=12.0,
        ),
        "hp_min_child_weight": sp.loguniform(lower=1e-5, upper=64.0),
        "hp_gamma": sp.loguniform(lower=1e-5, upper=64),
        "hp_lambda": sp.loguniform(lower=1e-5, upper=256),
        "hp_alpha": sp.loguniform(lower=1e-5, upper=256),
    }

    df["resource"] = 1  # Dummy fidelity

    bb_dict = {}
    for task in df.task.unique():
        df_task = df.loc[df.task == task]
        hypers = df_task[[col for col in df_task.columns if col.startswith("hp_")]]
        evals = df_task[[col for col in df_task.columns if col.startswith("metric_")]].values[:, None, None, :]
        bb_dict[task] = BlackboxTabular(
            hyperparameters=hypers,
            objectives_evaluations=evals,
            configuration_space=configuration_space,
            fidelity_space={'resource': randint(1, 1)},
            objectives_names=[
                col for col in df.columns if col.startswith("metric_")
            ],
        )
    serialize(
            bb_dict=bb_dict,
            path=repository_path / "icml-xgboost",
        )


class XGBoostRecipe(BlackboxRecipe):
    def __init__(self):
        super(XGBoostRecipe, self).__init__(
            name="icml-xgboost",
            cite_reference="A quantile-based approach for hyperparameter transfer learning."
            "Salinas, D., Shen, H., and Perrone, V. 2021.",
        )

    def _generate_on_disk(self):
        serialize_xgboost()


class DeepARRecipe(BlackboxRecipe):
    def __init__(self):
        super(DeepARRecipe, self).__init__(
            name="icml-deepar",
            cite_reference="A quantile-based approach for hyperparameter transfer learning."
            "Salinas, D., Shen, H., and Perrone, V. 2021.",
        )

    def _generate_on_disk(self):
        serialize_deepar()


if __name__ == "__main__":
    DeepARRecipe().generate(upload_on_hub=False)
    XGBoostRecipe().generate(upload_on_hub=False)
