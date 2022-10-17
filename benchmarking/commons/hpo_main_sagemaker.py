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
from typing import Optional, List
import logging

from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    default_sagemaker_session,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
import benchmarking
from benchmarking.commons.baselines import MethodArguments
from benchmarking.commons.hpo_main_local import (
    RealBenchmarkDefinitions,
    get_benchmark,
)
from benchmarking.commons.utils import sagemaker_estimator
from benchmarking.commons.hpo_main_common import (
    parse_args as _parse_args,
    get_metadata,
)
from benchmarking.commons.launch_remote_common import sagemaker_estimator_args


# SageMaker managed warm pools:
# https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html#train-warm-pools-resource-limits
# Maximum time a warm pool instance is kept alive, waiting to be associated with
# a new job. Setting this too large may lead to extra costs.
WARM_POOL_KEEP_ALIVE_PERIOD_IN_SECONDS = 10 * 60


def parse_args(methods: dict, extra_args: Optional[List[dict]] = None):
    if extra_args is None:
        extra_args = []
    else:
        extra_args = extra_args.copy()
    extra_args.extend(
        [
            dict(
                name="benchmark",
                type=str,
                default="resnet_cifar10",
                help="Benchmark to run",
            ),
            dict(
                name="max_failures",
                type=int,
                default=3,
                help=(
                    "Number of trials which can fail without experiment being "
                    "terminated"
                ),
            ),
            dict(
                name="warm_pool",
                type=int,
                default=0,
                help=(
                    "If 1, the SageMaker managed warm pools feature is used. "
                    "This can be more expensive, but also reduces startup "
                    "delays, leading to an experiment finishing in less time"
                ),
            ),
            dict(
                name="instance_type",
                type=str,
                help="AWS SageMaker instance type (overwrites default of benchmark)",
            ),
            dict(
                name="start_jobs_without_delay",
                type=int,
                default=0,
                help=(
                    "If 1, the tuner starts new trials immediately after "
                    "sending existing ones a stop signal. This leads to more "
                    "than n_workers instances being used during certain times, "
                    "which can lead to quotas being exceeded, or the warm pool "
                    "feature not working optimal."
                ),
            ),
        ]
    )
    args, method_names, seeds = _parse_args(methods, extra_args)
    args.warm_pool = bool(args.warm_pool)
    args.start_jobs_without_delay = bool(args.start_jobs_without_delay)
    return args, method_names, seeds


def main(
    methods: dict,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[callable] = None,
):
    args, method_names, seeds = parse_args(methods, extra_args)
    experiment_tag = args.experiment_tag
    benchmark_name = args.benchmark
    assert (
        len(method_names) == 1 and len(seeds) == 1
    ), "Can only launch single (method, seed). Use launch_remote to launch several combinations"
    method = method_names[0]
    seed = seeds[0]
    logging.getLogger().setLevel(logging.INFO)

    benchmark = get_benchmark(args, benchmark_definitions, sagemaker_backend=True)
    print(f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}")

    sm_args = sagemaker_estimator_args(
        entry_point=benchmark.script,
        experiment_tag="A",
        tuner_name="B",
        benchmark=benchmark,
    )
    del sm_args["checkpoint_s3_uri"]
    sm_args["sagemaker_session"] = default_sagemaker_session()
    sm_args["dependencies"] = benchmarking.__path__
    if args.warm_pool:
        print(
            "--------------------------------------------------------------------------\n"
            "Using SageMaker managed warm pools in order to decrease start-up delays.\n"
            f"In order for this to work, you need to have at least {benchmark.n_workers} quotas of the type\n"
            f"   {benchmark.instance_type} for training warm pool usage\n"
            "--------------------------------------------------------------------------"
        )
        sm_args["keep_alive_period_in_seconds"] = WARM_POOL_KEEP_ALIVE_PERIOD_IN_SECONDS
    if args.instance_type is not None:
        sm_args["instance_type"] = args.instance_type
    trial_backend = SageMakerBackend(
        sm_estimator=sagemaker_estimator[benchmark.framework](**sm_args),
        # names of metrics to track. Each metric will be detected by Sagemaker if it is written in the
        # following form: "[RMSE]: 1.2", see in train_main_example how metrics are logged for an example
        metrics_names=[benchmark.metric],
    )

    method_kwargs = {"max_resource_attr": benchmark.max_resource_attr}
    if extra_args is not None:
        assert map_extra_args is not None
        extra_args = map_extra_args(args)
        method_kwargs.update(extra_args)
    scheduler = methods[method](
        MethodArguments(
            config_space=benchmark.config_space,
            metric=benchmark.metric,
            mode=benchmark.mode,
            random_seed=seed,
            resource_attr=benchmark.resource_attr,
            verbose=True,
            **method_kwargs,
        )
    )

    stop_criterion = StoppingCriterion(
        max_wallclock_time=benchmark.max_wallclock_time,
        max_num_evaluations=benchmark.max_num_evaluations,
    )
    metadata = get_metadata(
        seed, method, experiment_tag, benchmark_name, benchmark, extra_args
    )
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=benchmark.n_workers,
        tuner_name=experiment_tag,
        metadata=metadata,
        save_tuner=args.save_tuner,
        sleep_time=5.0,
        max_failures=args.max_failures,
        start_jobs_without_delay=args.start_jobs_without_delay,
    )
    tuner.run()
