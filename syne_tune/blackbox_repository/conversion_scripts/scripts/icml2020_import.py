"""
Convert evaluations from
 A Quantile-based Approach for Hyperparameter Transfer Learning
 David Salinas Huibin Shen Valerio Perrone
 http://proceedings.mlr.press/v119/salinas20a/salinas20a.pdf
"""
from typing import Optional
import pandas as pd
import numpy as np
from syne_tune.blackbox_repository.blackbox_offline import serialize, BlackboxOffline
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
    upload,
)
import syne_tune.config_space as sp


def download(blackbox: str):
    import urllib

    root = "https://github.com/geoalgo/A-Quantile-based-Approach-for-Hyperparameter-Transfer-Learning/blob/master/src/blackbox/offline_evaluations/"
    urllib.request.urlretrieve(
        root + f"{blackbox}.csv.zip?raw=true", repository_path / f"{blackbox}.csv.zip"
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
        "hp_num_cells": sp.lograndint(lower=10, upper=10000),
        "hp_context_length_ratio": sp.loguniform(lower=0.05, upper=4),
    }

    serialize(
        {
            task: BlackboxOffline(
                df_evaluations=df.loc[df.task == task, :],
                configuration_space=configuration_space,
                objectives_names=[
                    col for col in df.columns if col.startswith("metric_")
                ],
            )
            for task in df.task.unique()
        },
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

    serialize(
        {
            task: BlackboxOffline(
                df_evaluations=df.loc[df.task == task, :],
                configuration_space=configuration_space,
                objectives_names=[
                    col for col in df.columns if col.startswith("metric_")
                ],
            )
            for task in df.task.unique()
        },
        path=repository_path / "icml-xgboost",
    )


def generate_deepar(s3_root: Optional[str] = None):
    serialize_deepar()
    upload(name="icml-deepar", s3_root=s3_root)


def generate_xgboost(s3_root: Optional[str] = None):
    serialize_xgboost()
    upload(name="icml-xgboost", s3_root=s3_root)


if __name__ == "__main__":
    generate_deepar()
    generate_xgboost()
