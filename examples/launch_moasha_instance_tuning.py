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
Example showing how to tune instance types and hyperparameters with a Sagemaker Framework.
"""
import logging
from pathlib import Path

from sagemaker.huggingface import HuggingFace

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.instance_info import select_instance_type
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.config_space import loguniform, choice
from syne_tune.constants import (
    ST_WORKER_TIME,
    ST_WORKER_COST,
    ST_INSTANCE_TYPE,
)
from syne_tune.optimizer.schedulers.multiobjective import MOASHA
from syne_tune.remote.estimators import (
    DEFAULT_CPU_INSTANCE_SMALL,
    HUGGINGFACE_LATEST_FRAMEWORK_VERSION,
    HUGGINGFACE_LATEST_TRANSFORMERS_VERSION,
    HUGGINGFACE_LATEST_PYTORCH_VERSION,
    HUGGINGFACE_LATEST_PY_VERSION,
)
from syne_tune.remote.remote_launcher import RemoteLauncher

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    n_workers = 2
    epochs = 4

    # Select the instance types that are searched.
    # Alternatively, you can define the instance list explicitly:
    # :code:`instance_types = ["ml.c5.xlarge", "ml.m5.2xlarge"]`
    instance_types = select_instance_type(min_gpu=1, max_cost_per_hour=5.0)

    print(f"tuning over hyperparameters and instance types: {instance_types}")

    # define a search space that contains hyperparameters (learning-rate, weight-decay) and instance-type.
    config_space = {
        ST_INSTANCE_TYPE: choice(instance_types),
        "learning_rate": loguniform(1e-6, 1e-4),
        "weight_decay": loguniform(1e-5, 1e-2),
        "epochs": epochs,
        "dataset_path": "./",
    }
    entry_point = (
        Path(__file__).parent.parent
        / "benchmarking"
        / "training_scripts"
        / "distilbert_on_imdb"
        / "distilbert_on_imdb.py"
    )
    metric = "accuracy"

    # Define a MOASHA scheduler that searches over the config space to maximise accuracy and minimize cost and time.
    scheduler = MOASHA(
        max_t=epochs,
        time_attr="step",
        metrics=[metric, ST_WORKER_COST, ST_WORKER_TIME],
        mode=["max", "min", "min"],
        config_space=config_space,
    )

    # Define the training function to be tuned, use the Sagemaker backend to execute trials as separate training job
    # (since they are quite expensive).
    trial_backend = SageMakerBackend(
        sm_estimator=HuggingFace(
            framework_version=HUGGINGFACE_LATEST_FRAMEWORK_VERSION,
            transformers_version=HUGGINGFACE_LATEST_TRANSFORMERS_VERSION,
            pytorch_version=HUGGINGFACE_LATEST_PYTORCH_VERSION,
            py_version=HUGGINGFACE_LATEST_PY_VERSION,
            entry_point=str(entry_point),
            base_job_name="hpo-transformer",
            # instance-type given here are override by Syne Tune with values sampled from ST_INSTANCE_TYPE.
            instance_type=DEFAULT_CPU_INSTANCE_SMALL,
            instance_count=1,
            max_run=3600,
            role=get_execution_role(),
            dependencies=[str(Path(__file__).parent.parent / "benchmarking")],
            sagemaker_session=default_sagemaker_session(),
            disable_profiler=True,
            debugger_hook_config=False,
        ),
    )

    remote_launcher = RemoteLauncher(
        tuner=Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=StoppingCriterion(max_wallclock_time=3600, max_cost=10.0),
            n_workers=n_workers,
            sleep_time=5.0,
        ),
        dependencies=[str(Path(__file__).parent.parent / "benchmarking")],
    )

    remote_launcher.run(wait=False)
