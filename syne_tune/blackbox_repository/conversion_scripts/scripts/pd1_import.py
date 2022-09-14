# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import json
import logging
import tarfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from syne_tune import config_space
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts import (
    default_metric,
    metric_elapsed_time,
    resource_attr,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
    download_file,
)
from syne_tune.blackbox_repository.serialize import (
    deserialize_configspace,
    deserialize_metadata,
    serialize_configspace,
    serialize_metadata,
)
from syne_tune.config_space import loguniform, randint, uniform
from syne_tune.util import catchtime

logger = logging.getLogger(__name__)

BLACKBOX_NAME = "pd1"

METRIC_VALID_ERROR = "metric_valid_error_rate"

METRIC_ELAPSED_TIME = "metric_elapsed_time"

RESOURCE_ATTR = "global_step"

CONFIGURATION_SPACE = {
    "lr_initial_value": loguniform(1e-5, 10),
    "lr_power": uniform(0.1, 2.0),
    "lr_decay_steps_factor": uniform(0.01, 0.99),
    "one_minus_momentum": loguniform(1e-3, 1.0),
}

COLUMN_RENAMING = {
    "hps.lr_hparams.initial_value": "lr_initial_value",
    "hps.lr_hparams.power": "lr_power",
    "hps.lr_hparams.decay_steps_factor": "lr_decay_steps_factor",
    "hps.opt_hparams.momentum": "one_minus_momentum",
    "valid/ce_loss": "metric_valid_ce_loss",
    "valid/error_rate": METRIC_VALID_ERROR,
    "epoch": "epoch",
    "eval_time": METRIC_ELAPSED_TIME,
    "global_step": RESOURCE_ATTR,
}


def convert_task(task_data):
    hyperparameters = task_data[list(CONFIGURATION_SPACE.keys())]

    objective_names = [
        "metric_valid_error_rate",
        "metric_valid_ce_loss",
        METRIC_ELAPSED_TIME,
    ]
    available_objectives = [
        objective_name
        for objective_name, is_not_available in task_data[objective_names]
        .isnull()
        .all()
        .to_dict()
        .items()
        if not is_not_available
    ]
    task_data.insert(
        0,
        "num_steps",
        task_data[available_objectives[0]].map(lambda x: 0 if x is None else len(x)),
    )
    learning_curve_length = task_data["num_steps"].max()

    def pad_with_nans(learning_curve, length):
        if learning_curve is None:
            return length * [np.nan]
        return learning_curve + (length - len(learning_curve)) * [np.nan]

    objectives_evaluations = list()
    for o in available_objectives:
        task_data[o] = task_data[o].apply(pad_with_nans, args=(learning_curve_length,))
        objectives_evaluations.append(np.array(task_data[o].to_list()))
    objectives_evaluations = np.expand_dims(
        np.stack(objectives_evaluations, axis=-1), 1
    )

    fidelity_space = {RESOURCE_ATTR: randint(lower=1, upper=learning_curve_length)}

    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=CONFIGURATION_SPACE,
        fidelity_space=fidelity_space,
        objectives_evaluations=objectives_evaluations,
        fidelity_values=np.arange(1, learning_curve_length + 1),
        objectives_names=available_objectives,
    )


class PD1Recipe(BlackboxRecipe):
    def __init__(self):
        super(PD1Recipe, self).__init__(
            name=BLACKBOX_NAME,
            cite_reference="Pre-trained Gaussian processes for Bayesian optimization. "
            "Wang, Z. and Dahl G. and Swersky K. and Lee C. and Mariet Z. and Nado Z. and Gilmer J. and Snoek J. and "
            "Ghahramani Z. 2021.",
        )

    def _download_data(self):
        file_name = repository_path / f"{BLACKBOX_NAME}.tar.gz"
        if not file_name.exists():
            logger.info(f"Did not find {file_name}. Starting download.")
            download_file(
                "http://storage.googleapis.com/gresearch/pint/pd1.tar.gz", file_name
            )
        else:
            logger.info(f"Skip downloading since {file_name} is available locally.")

    def _convert_data(self) -> Dict[str, BlackboxTabular]:
        with tarfile.open(repository_path / f"{BLACKBOX_NAME}.tar.gz") as f:
            f.extractall(path=repository_path)
        data = []
        for matched in ["matched", "unmatched"]:
            path = (
                repository_path
                / BLACKBOX_NAME
                / f"pd1_{matched}_phase1_results.jsonl.gz"
            )
            with open(path, "rb") as fin:
                data.append(
                    pd.read_json(fin, orient="records", lines=True, compression="gzip")
                )
        df = pd.concat(data)
        df["eval_time"] = df["eval_time"].apply(
            lambda x: None if x is None else np.cumsum(x).tolist()
        )

        tasks = df[
            ["dataset", "model", "hps.batch_size", "hps.activation_fn"]
        ].drop_duplicates()
        bb_dict = {}
        for _, task in tasks.iterrows():
            activation_name = (
                ""
                if task["hps.activation_fn"] is None
                else f"_{task['hps.activation_fn']}"
            )
            task_name = "{}_{}{}_batch_size_{}".format(
                task["dataset"],
                task["model"],
                activation_name,
                task["hps.batch_size"],
            )
            task_data = df[
                (df["dataset"] == task["dataset"])
                & (df["model"] == task["model"])
                & (df["hps.batch_size"] == task["hps.batch_size"])
            ]
            if task["hps.activation_fn"] is not None:
                task_data = task_data[
                    task_data["hps.activation_fn"] == task["hps.activation_fn"]
                ]
            task_data = task_data.reset_index()
            task_data = task_data[list(COLUMN_RENAMING)]
            task_data.columns = list(COLUMN_RENAMING.values())
            with catchtime(f"converting task {task_name}"):
                bb_dict[task_name] = convert_task(task_data)
        return bb_dict

    def _save_data(self, bb_dict: Dict[str, BlackboxTabular]) -> None:
        with catchtime("saving to disk"):
            serialize(
                bb_dict=bb_dict,
                path=repository_path / BLACKBOX_NAME,
                metadata={
                    metric_elapsed_time: METRIC_ELAPSED_TIME,
                    default_metric: METRIC_VALID_ERROR,
                    resource_attr: RESOURCE_ATTR,
                },
            )

    def _generate_on_disk(self):
        matched_file = (
            repository_path / BLACKBOX_NAME / "pd1_matched_phase1_results.jsonl.gz"
        )
        unmatched_file = (
            repository_path / BLACKBOX_NAME / "pd1_unmatched_phase1_results.jsonl.gz"
        )
        if matched_file.exists() and unmatched_file.exists():
            return
        self._download_data()
        bb_dict = self._convert_data()
        self._save_data(bb_dict)


def serialize(
    bb_dict: Dict[str, BlackboxTabular], path: str, metadata: Optional[Dict] = None
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

        with open(path / f"{task}-fidelity_space.json", "w") as f:
            json.dump(
                {
                    k: config_space.to_dict(v)
                    for k, v in bb_dict[task].fidelity_space.items()
                },
                f,
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


def deserialize(path: str) -> Dict[str, BlackboxTabular]:
    """
    Deserialize blackboxes contained in a path that were saved with `serialize` above.
    TODO: the API is currently dissonant with `serialize`, `deserialize` for BlackboxOffline as `serialize` is there a member.
    A possible way to unify is to have serialize also be a free function for BlackboxOffline.
    :param path: a path that contains blackboxes that were saved with `serialize`
    :return: a dictionary from task name to blackbox
    """
    path = Path(path)
    configuration_space, _ = deserialize_configspace(path)
    metadata = deserialize_metadata(path)
    objectives_names = metadata["objectives_names"]
    task_names = metadata["task_names"]

    bb_dict = {}
    for task in task_names:
        hyperparameters = pd.read_parquet(
            Path(path) / f"{task}-hyperparameters.parquet", engine="fastparquet"
        )
        with open(path / f"{task}-fidelity_space.json", "r") as file:
            fidelity_space = {
                k: config_space.from_dict(v) for k, v in json.load(file).items()
            }

        with open(path / f"{task}-fidelity_values.npy", "rb") as f:
            fidelity_values = np.load(f)

        with open(path / f"{task}-objectives_evaluations.npy", "rb") as f:
            objectives_evaluations = np.load(f)

        bb_dict[task] = BlackboxTabular(
            hyperparameters=hyperparameters,
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            objectives_evaluations=objectives_evaluations,
            fidelity_values=fidelity_values,
            objectives_names=objectives_names,
        )
    return bb_dict


if __name__ == "__main__":
    PD1Recipe().generate()
