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
import os
import logging
from pathlib import Path
from argparse import ArgumentParser

from syne_tune.backend import LocalBackend, SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.optimizer.baselines import (
    ASHA,
)

from syne_tune import Tuner, StoppingCriterion
from syne_tune.remote.constants import (
    DEFAULT_CPU_INSTANCE_SMALL,
    PYTORCH_LATEST_FRAMEWORK,
    PYTORCH_LATEST_PY_VERSION,
)
from examples.training_scripts.height_example.train_height_config_json import (
    height_config_space,
    RESOURCE_ATTR,
    METRIC_ATTR,
    METRIC_MODE,
    MAX_RESOURCE_ATTR,
)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--use_sagemaker_backend", type=int, default=0)
    args = parser.parse_args()
    use_sagemaker_backend = bool(args.use_sagemaker_backend)

    random_seed = 31415927
    max_epochs = 100
    n_workers = 4
    max_wallclock_time = 5 * 60 if use_sagemaker_backend else 10

    config_space = height_config_space(max_epochs)
    entry_point = (
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height_config_json.py"
    )

    scheduler = ASHA(
        config_space,
        metric=METRIC_ATTR,
        mode=METRIC_MODE,
        max_resource_attr=MAX_RESOURCE_ATTR,
        resource_attr=RESOURCE_ATTR,
    )

    if not use_sagemaker_backend:
        trial_backend = LocalBackend(
            entry_point=str(entry_point),
            pass_args_as_json=True,
        )
    else:
        from sagemaker.pytorch import PyTorch
        import syne_tune

        if "AWS_DEFAULT_REGION" not in os.environ:
            os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
        trial_backend = SageMakerBackend(
            sm_estimator=PyTorch(
                entry_point=str(entry_point),
                instance_type=DEFAULT_CPU_INSTANCE_SMALL,
                instance_count=1,
                framework_version=PYTORCH_LATEST_FRAMEWORK,
                py_version=PYTORCH_LATEST_PY_VERSION,
                role=get_execution_role(),
                dependencies=syne_tune.__path__,
                max_run=10 * 60,
                sagemaker_session=default_sagemaker_session(),
                disable_profiler=True,
                debugger_hook_config=False,
                keep_alive_period_in_seconds=60,  # warm pool feature
            ),
            metrics_names=[METRIC_ATTR],
            pass_args_as_json=True,
        )

    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        start_jobs_without_delay=False,
    )

    tuner.run()
