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
"""
This launches an HPO tuning several hyperparameters of a gluonts model.
To run this example locally, you need to have installed dependencies in `requirements.txt` in your current interpreter.
"""
import logging
from pathlib import Path

import numpy as np

from sagemaker.mxnet import MXNet

from syne_tune.backend import LocalBackend, SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import loguniform, lograndint


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    epochs = 50

    config_space = {
        "lr": loguniform(1e-4, 1e-1),
        "epochs": epochs,
        "num_cells": lograndint(lower=1, upper=80),
        "num_layers": lograndint(lower=1, upper=10),
        "dataset": "electricity"
        # "dataset": "m4_hourly"
    }

    mode = "min"
    metric = "mean_wQuantileLoss"
    entry_point = Path(__file__).parent / "training_scripts" / "gluonts" / "train_gluonts.py"

    evaluate_trials_on_sagemaker = False

    if evaluate_trials_on_sagemaker:
        # evaluate trials on Sagemaker
        trial_backend = SageMakerBackend(
            sm_estimator=MXNet(
                entry_point=entry_point.name,
                source_dir=str(entry_point.parent),
                instance_type="ml.c5.2xlarge",
                instance_count=1,
                role=get_execution_role(),
                max_run=10 * 60,
                framework_version='1.7',
                py_version='py3',
                base_job_name='hpo-gluonts',
            ),
            # names of metrics to track. Each metric will be detected by Sagemaker if it is written in the
            # following form: "[RMSE]: 1.2", see in train_main_example how metrics are logged for an example
            metrics_names=[metric],
        )
    else:
        # evaluate trials locally, replace with SagemakerBackend to evaluate trials on Sagemaker
        trial_backend = LocalBackend(entry_point=str(entry_point))

    # see examples to see other schedulers, mobster, Raytune, multiobjective, etc...
    scheduler = ASHA(
        config_space,
        max_t=epochs,
        resource_attr='epoch_no',
        mode='min',
        metric=metric
    )

    wallclock_time_budget = 3600 if evaluate_trials_on_sagemaker else 600
    dollar_cost_budget = 20.0

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        # stops if wallclock time or dollar-cost exceeds budget,
        # dollar-cost is only available when running on Sagemaker
        stop_criterion=StoppingCriterion(max_wallclock_time=wallclock_time_budget, max_cost=dollar_cost_budget),
        n_workers=4,
        # some failures may happen when SGD diverges with NaNs
        max_failures=10,
    )

    # launch the tuning
    tuner.run()