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
from argparse import ArgumentParser
import os
import boto3

from sagemaker.pytorch import PyTorch

from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        required=True,
        help="number of seeds to run",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="first seed to run",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="number of parallel workers",
    )
    parser.add_argument(
        "--max_wallclock_time",
        type=int,
        default=3 * 3600,
        help="maximum wallclock time of experiment",
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    suffix = random_string(4)

    if boto3.Session().region_name is None:
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    environment = {"AWS_DEFAULT_REGION": boto3.Session().region_name}

    for seed in tqdm(range(args.start_seed, args.num_seeds)):
        checkpoint_s3_uri = s3_experiment_path(
            tuner_name=str(seed), experiment_name=experiment_tag
        )
        sm_args = dict(
            entry_point="launch_experiment.py",
            source_dir=str(Path(__file__).parent),
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type="ml.c5.4xlarge",
            instance_count=1,
            py_version="py3",
            framework_version="1.7.1",
            max_run=3600 * 72,
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
            environment=environment,
        )

        sm_args["hyperparameters"] = {
            "experiment_tag": experiment_tag,
            "seed": seed,
            "n_workers": args.n_workers,
            "max_wallclock_time": args.max_wallclock_time,
        }
        print(
            f"{experiment_tag}-{seed}\n"
            f"hyperparameters = {sm_args['hyperparameters']}\n"
            f"Results written to {checkpoint_s3_uri}"
        )
        est = PyTorch(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{seed}-{suffix}", wait=False)

    print(
        "\nLaunched all requested experiments. Once everything is done, use this "
        "command to sync result files from S3:\n"
        f"$ aws s3 sync {s3_experiment_path(experiment_name=experiment_tag)}/ "
        f'~/syne-tune/{experiment_tag}/ --exclude "*" '
        '--include "*metadata.json" --include "*results.csv.zip"'
    )
