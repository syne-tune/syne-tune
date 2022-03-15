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
Example showing how to run on Sagemaker with a Sagemaker Framework.
"""
import logging
from pathlib import Path

from sagemaker.pytorch import PyTorch

from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner
from syne_tune.config_space import randint
from syne_tune import StoppingCriterion


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100)
    }
    entry_point = Path(__file__).parent / "training_scripts" / "height_example" / "train_height.py"
    mode = "min"
    metric = "mean_loss"

    # Random search without stopping
    scheduler = RandomSearch(
        config_space,
        mode=mode,
        metric=metric,
        random_seed=random_seed)

    trial_backend = SageMakerBackend(
        # we tune a PyTorch Framework from Sagemaker
        sm_estimator=PyTorch(
            entry_point=str(entry_point),
            instance_type="ml.m5.large",
            instance_count=1,
            role=get_execution_role(),
            max_run=10 * 60,
            framework_version='1.7.1',
            py_version='py3',
        ),
        # names of metrics to track. Each metric will be detected by Sagemaker if it is written in the
        # following form: "[RMSE]: 1.2", see in train_main_example how metrics are logged for an example
        metrics_names=[metric],
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=600)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=5.0,
        tuner_name="hpo-hyperband",
    )

    tuner.run()
