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
import logging
from argparse import ArgumentParser

from sagemaker.pytorch import PyTorch

from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.optimizer.baselines import (
    ASHA,
    RandomSearch,
    BayesianOptimization,
    MOBSTER,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from benchmarking.definitions.definition_resnet_cifar10 import (
    resnet_cifar10_benchmark,
)
from syne_tune.util import repository_root_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed (for repetitions)",
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
    parser.add_argument(
        "--method",
        type=str,
        choices=["asha", "mobster", "rs", "bo"],
        default="asha",
        help="method to run",
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag

    params = {
        "backend": "sagemaker",
        "dataset_path": "./",
        "num_gpus": 1,
        "max_resource_level": 27,
        "instance_type": "ml.g4dn.xlarge",
    }
    benchmark = resnet_cifar10_benchmark(params)
    logging.getLogger().setLevel(logging.INFO)

    print(f"Starting experiment ({args.seed}) of {experiment_tag}")

    script_path = benchmark["script"]
    trial_backend = SageMakerBackend(
        # we tune a PyTorch Framework from Sagemaker
        sm_estimator=PyTorch(
            entry_point=script_path.name,
            source_dir=str(script_path.parent),
            instance_type=params["instance_type"],
            instance_count=1,
            role=get_execution_role(),
            framework_version="1.7.1",
            py_version="py3",
            max_run=2 * args.max_wallclock_time,
            dependencies=[str(repository_root_path() / "benchmarking/")],
            disable_profiler=True,
            debugger_hook_config=False,
            sagemaker_session=default_sagemaker_session(),
        ),
        # names of metrics to track. Each metric will be detected by Sagemaker if it is written in the
        # following form: "[RMSE]: 1.2", see in train_main_example how metrics are logged for an example
        metrics_names=[benchmark["metric"]],
    )

    common_kwargs = dict(
        search_options={"debug_log": True},
        metric=benchmark["metric"],
        mode=benchmark["mode"],
        max_resource_attr=benchmark["max_resource_attr"],
        random_seed=args.seed,
    )
    if args.method == "asha":
        scheduler = ASHA(
            benchmark["config_space"],
            resource_attr=benchmark["resource_attr"],
            **common_kwargs,
        )
    elif args.method == "mobster":
        scheduler = MOBSTER(
            benchmark["config_space"],
            resource_attr=benchmark["resource_attr"],
            **common_kwargs,
        )
    elif args.method == "rs":
        scheduler = RandomSearch(benchmark["config_space"], **common_kwargs)
    else:
        assert args.method == "bo"
        scheduler = BayesianOptimization(benchmark["config_space"], **common_kwargs)

    stop_criterion = StoppingCriterion(
        max_wallclock_time=args.max_wallclock_time,
    )
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=args.n_workers,
        sleep_time=5.0,
        max_failures=3,
        tuner_name=experiment_tag,
        metadata={
            "seed": args.seed,
            "algorithm": args.method,
            "type": "stopping",
            "tag": experiment_tag,
            "benchmark": "resnet_cifar10",
            "n_workers": args.n_workers,
            "max_wallclock_time": args.max_wallclock_time,
        },
    )

    tuner.run()
