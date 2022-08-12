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
import logging
import tarfile

import numpy as np
import pandas as pd

from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular, serialize
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
    "train/ce_loss": "metric_train_ce_loss",
    "valid/ce_loss": "metric_valid_ce_loss",
    "test/ce_loss": "metric_test_ce_loss",
    "train/error_rate": "metric_train_error_rate",
    "valid/error_rate": METRIC_VALID_ERROR,
    "test/error_rate": "metric_test_error_rate",
    "epoch": "epoch",
    "eval_time": METRIC_ELAPSED_TIME,
    "global_step": RESOURCE_ATTR,
}


def convert_task(task_data):
    hyperparameters = task_data[list(CONFIGURATION_SPACE.keys())]

    objective_names = [
        "metric_train_ce_loss",
        "metric_valid_ce_loss",
        "metric_test_ce_loss",
        "metric_train_error_rate",
        "metric_valid_error_rate",
        "metric_test_error_rate",
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
    # global_step = task_data.iloc[task_data["num_steps"].idxmax()]["global_step"]
    # epoch = task_data.iloc[task_data["num_steps"].idxmax()]["epoch"]
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

    def _generate_on_disk(self):
        # Download data
        file_name = repository_path / f"{BLACKBOX_NAME}.tar.gz"
        if not file_name.exists():
            logger.info(f"Did not find {file_name}. Starting download.")
            download_file(
                "http://storage.googleapis.com/gresearch/pint/pd1.tar.gz", file_name
            )
            with tarfile.open(file_name) as f:
                f.extractall(path=repository_path)
        else:
            logger.info(f"Skip downloading since {file_name} is available locally.")

        # Convert data
        data = []
        for matched in ["matched"]:  # , "unmatched"]: #TODO:
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
        df["eval_time"].apply(lambda x: None if x is None else np.cumsum)

        tasks = df[
            ["dataset", "model", "hps.batch_size", "hps.activation_fn"]
        ].drop_duplicates()
        bb_dict = {}
        for _, task in tasks.iterrows():
            task_name = "{}_{}{}_batch_size{}".format(
                task["dataset"],
                task["model"],
                ""
                if task["hps.activation_fn"] is None
                else f"_{task['hps.activation_fn']}",
                task["hps.batch_size"],
            )
            task_data = df[
                (df["dataset"] == task["dataset"])
                & (df["model"] == task["model"])
                & (df["hps.batch_size"] == task["hps.batch_size"])
            ]
            if task["hps.activation_fn"] is not None:
                task_data = task_data[
                    df["hps.activation_fn"] == task["hps.activation_fn"]
                ]
            task_data = task_data.reset_index()
            task_data = task_data[list(COLUMN_RENAMING)]
            task_data.columns = list(COLUMN_RENAMING.values())
            with catchtime(f"converting task {task_name}"):
                bb_dict[task_name] = convert_task(task_data)

        # Save converted data
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
