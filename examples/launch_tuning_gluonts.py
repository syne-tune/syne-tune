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
To run this example locally, you need to have installed dependencies in ``requirements.txt`` in your current interpreter.
"""
import logging
from pathlib import Path

import numpy as np
from sagemaker.mxnet import MXNet

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend, SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.config_space import loguniform, lograndint
from syne_tune.optimizer.baselines import ASHA
from syne_tune.remote.estimators import (
    DEFAULT_CPU_INSTANCE,
    MXNET_LATEST_VERSION,
    MXNET_LATEST_PY_VERSION,
)

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    epochs = 50
    n_workers = 4
    mode = "min"
    metric = "mean_wQuantileLoss"
    max_resource_attr = "epochs"

    config_space = {
        "lr": loguniform(1e-4, 1e-1),
        "num_cells": lograndint(lower=1, upper=80),
        "num_layers": lograndint(lower=1, upper=10),
        max_resource_attr: epochs,
        "dataset": "electricity"
        # "dataset": "m4_hourly"
    }

    entry_point = (
        Path(__file__).parent / "training_scripts" / "gluonts" / "train_gluonts.py"
    )

    # Note: In order to run this locally (value False), you need to have GluonTS and its
    # dependencies installed
    evaluate_trials_on_sagemaker = True

    if evaluate_trials_on_sagemaker:
        # Evaluate trials on Sagemaker
        trial_backend = SageMakerBackend(
            sm_estimator=MXNet(
                framework_version=MXNET_LATEST_VERSION,
                py_version=MXNET_LATEST_PY_VERSION,
                entry_point=entry_point.name,
                source_dir=str(entry_point.parent),
                instance_type=DEFAULT_CPU_INSTANCE,
                instance_count=1,
                role=get_execution_role(),
                max_run=10 * 60,
                base_job_name="hpo-gluonts",
                sagemaker_session=default_sagemaker_session(),
                disable_profiler=True,
                debugger_hook_config=False,
            ),
            metrics_names=[metric],
        )
    else:
        # evaluate trials locally, replace with SageMakerBackend to evaluate trials on Sagemaker
        trial_backend = LocalBackend(entry_point=str(entry_point))

    # Use asynchronous successive halving (ASHA)
    scheduler = ASHA(
        config_space,
        metric=metric,
        max_resource_attr=max_resource_attr,
        resource_attr="epoch_no",
    )

    max_wallclock_time = (
        3000 if evaluate_trials_on_sagemaker else 600
    )  # wall clock time can be increased to 1 hour for more performance
    dollar_cost_budget = 20.0

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        # stops if wallclock time or dollar-cost exceeds budget,
        # dollar-cost is only available when running on Sagemaker
        stop_criterion=StoppingCriterion(
            max_wallclock_time=max_wallclock_time, max_cost=dollar_cost_budget
        ),
        n_workers=n_workers,
        # some failures may happen when SGD diverges with NaNs
        max_failures=10,
    )

    # launch the tuning
    tuner.run()
