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
from argparse import ArgumentParser
from pathlib import Path

from coolname import generate_slug

import benchmarking
import syne_tune
from benchmarking.nursery.benchmark_automl.baselines import methods, Methods
from syne_tune.remote.estimators import (
    basic_cpu_instance_sagemaker_estimator,
)
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.util import s3_experiment_path, random_string

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default=generate_slug(2)
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    hash = random_string(4)

    for method in methods.keys():
        sm_args = dict(
            entry_point="benchmark_main.py",
            source_dir=str(Path(__file__).parent),
            # instance_type="local",
            checkpoint_s3_uri=s3_experiment_path(
                tuner_name=method, experiment_name=experiment_tag
            ),
            max_run=3600 * 72,
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
        )

        if method != Methods.MOBSTER:
            print(f"{experiment_tag}-{method}")
            sm_args["hyperparameters"] = {
                "experiment_tag": experiment_tag,
                "num_seeds": 30,
                "method": method,
            }
            est = basic_cpu_instance_sagemaker_estimator(**sm_args)
            est.fit(job_name=f"{experiment_tag}-{method}-{hash}", wait=False)
        else:
            # For mobster, we schedule one job per seed as the method takes much longer
            for seed in range(30):
                print(f"{experiment_tag}-{method}-{seed}")
                sm_args["hyperparameters"] = {
                    "experiment_tag": experiment_tag,
                    "num_seeds": seed,
                    "run_all_seed": 0,
                    "method": method,
                }
                est = basic_cpu_instance_sagemaker_estimator(**sm_args)
                est.fit(job_name=f"{experiment_tag}-{method}-{seed}-{hash}", wait=False)
