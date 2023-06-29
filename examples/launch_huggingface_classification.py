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
Example for how to fine-tune a DistilBERT model on the IMDB sentiment classification task using the Hugging Face SageMaker Framework.
"""
import logging
from pathlib import Path

from sagemaker.huggingface import HuggingFace

import syne_tune
from benchmarking.benchmark_definitions import distilbert_imdb_benchmark
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.remote.constants import (
    HUGGINGFACE_LATEST_FRAMEWORK_VERSION,
    HUGGINGFACE_LATEST_PYTORCH_VERSION,
    HUGGINGFACE_LATEST_TRANSFORMERS_VERSION,
    HUGGINGFACE_LATEST_PY_VERSION,
)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # We pick the DistilBERT on IMDB benchmark
    # The 'benchmark' dict contains arguments needed by scheduler and
    # searcher (e.g., 'mode', 'metric'), along with suggested default values
    # for other arguments (which you are free to override)
    random_seed = 31415927
    n_workers = 4
    benchmark = distilbert_imdb_benchmark()
    mode = benchmark.mode
    metric = benchmark.metric
    config_space = benchmark.config_space

    # Define Hugging Face SageMaker estimator
    root = Path(syne_tune.__path__[0]).parent
    estimator = HuggingFace(
        framework_version=HUGGINGFACE_LATEST_FRAMEWORK_VERSION,
        transformers_version=HUGGINGFACE_LATEST_TRANSFORMERS_VERSION,
        pytorch_version=HUGGINGFACE_LATEST_PYTORCH_VERSION,
        py_version=HUGGINGFACE_LATEST_PY_VERSION,
        entry_point=str(benchmark.script),
        base_job_name="hpo-transformer",
        instance_type=benchmark.instance_type,
        instance_count=1,
        role=get_execution_role(),
        dependencies=[root / "benchmarking"],
        sagemaker_session=default_sagemaker_session(),
    )

    # SageMaker backend
    trial_backend = SageMakerBackend(
        sm_estimator=estimator,
        metrics_names=[metric],
    )

    # Random search without stopping
    scheduler = RandomSearch(
        config_space, mode=mode, metric=metric, random_seed=random_seed
    )

    stop_criterion = StoppingCriterion(
        max_wallclock_time=3000
    )  # wall clock time can be increased to 1 hour for more performance
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
