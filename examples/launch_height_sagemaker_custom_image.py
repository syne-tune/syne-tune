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
Example showing how to run on Sagemaker with a custom docker image.
"""
import logging
from pathlib import Path

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.custom_framework import CustomFramework
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.config_space import randint
from examples.training_scripts.height_example.train_height import (
    METRIC_ATTR,
    METRIC_MODE,
    MAX_RESOURCE_ATTR,
)
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.remote.constants import DEFAULT_CPU_INSTANCE_SMALL

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        MAX_RESOURCE_ATTR: max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    # Random search without stopping
    scheduler = RandomSearch(
        config_space, mode=METRIC_MODE, metric=METRIC_ATTR, random_seed=random_seed
    )

    # indicate here an image_uri that is available in ecr, something like that "XXXXXXXXXXXX.dkr.ecr.us-west-2.amazonaws.com/my_image:latest"
    image_uri = ...

    trial_backend = SageMakerBackend(
        sm_estimator=CustomFramework(
            entry_point=entry_point,
            instance_type=DEFAULT_CPU_INSTANCE_SMALL,
            instance_count=1,
            role=get_execution_role(),
            image_uri=image_uri,
            max_run=10 * 60,
            job_name_prefix="hpo-hyperband",
            sagemaker_session=default_sagemaker_session(),
            disable_profiler=True,
            debugger_hook_config=False,
        ),
        # names of metrics to track. Each metric will be detected by Sagemaker if it is written in the
        # following form: "[RMSE]: 1.2", see in train_main_example how metrics are logged for an example
        metrics_names=[METRIC_ATTR],
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=600)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=5.0,
    )

    tuner.run()
