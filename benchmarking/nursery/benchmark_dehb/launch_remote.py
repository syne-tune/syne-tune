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
from pathlib import Path
from tqdm import tqdm

from sagemaker.pytorch import PyTorch

from benchmarking.nursery.benchmark_automl.benchmark_main import parse_args
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


if __name__ == "__main__":
    from benchmarking.nursery.benchmark_dehb.baselines import methods
    from benchmarking.nursery.benchmark_dehb.benchmark_definitions import (
        benchmark_definitions,
    )

    args, method_names, benchmark_names, _ = parse_args(
        methods,
        benchmark_definitions,
    )
    if len(benchmark_names) == 1:
        benchmark_name = benchmark_names[0]
    else:
        benchmark_name = None
    experiment_tag = args.experiment_tag
    suffix = random_string(4)

    for method in tqdm(method_names):
        checkpoint_s3_uri = s3_experiment_path(
            tuner_name=method, experiment_name=experiment_tag
        )
        sm_args = dict(
            entry_point="benchmark_main.py",
            source_dir=str(Path(__file__).parent),
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type="ml.c5.4xlarge",
            instance_count=1,
            py_version="py38",
            framework_version="1.10.0",
            max_run=3600 * 72,
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
        )

        hyperparameters = {
            "experiment_tag": experiment_tag,
            "num_seeds": args.num_seeds,
            "method": method,
        }
        if benchmark_name is not None:
            hyperparameters["benchmark"] = benchmark_name
        if args.num_brackets is not None:
            hyperparameters["num_brackets"] = args.num_brackets
        if args.start_seed is not None:
            hyperparameters["start_seed"] = args.start_seed
        print(
            f"{experiment_tag}-{method}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {checkpoint_s3_uri}"
        )
        sm_args["hyperparameters"] = hyperparameters
        est = PyTorch(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{method}-{suffix}", wait=False)

    print(
        "\nLaunched all requested experiments. Once everything is done, use this "
        "command to sync result files from S3:\n"
        f"$ aws s3 sync {s3_experiment_path(experiment_name=experiment_tag)}/ "
        f'~/syne-tune/{experiment_tag}/ --exclude "*" '
        '--include "*metadata.json" --include "*results.csv.zip"'
    )
