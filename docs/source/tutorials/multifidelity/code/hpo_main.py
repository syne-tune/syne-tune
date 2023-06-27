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

from syne_tune.experiments.benchmark_definitions import nas201_benchmark
from syne_tune.backend.simulator_backend.simulator_callback import (
    SimulatorCallback,
)
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.baselines import (
    ASHA,
    MOBSTER,
    HyperTune,
    SyncHyperband,
    SyncBOHB,
    SyncMOBSTER,
    DEHB,
)
from syne_tune import Tuner, StoppingCriterion

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        choices=(
            "ASHA-STOP",
            "ASHA-PROM",
            "ASHA6-STOP",
            "MOBSTER-JOINT",
            "MOBSTER-INDEP",
            "HYPERTUNE-INDEP",
            "HYPERTUNE4-INDEP",
            "HYPERTUNE-JOINT",
            "SYNCHB",
            "SYNCSH",
            "SYNCMOBSTER",
            "BOHB",
            "DEHB",
        ),
        default="ASHA-STOP",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=31415927,
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="mf-tutorial",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("cifar10", "cifar100", "ImageNet16-120"),
        default="cifar100",
    )
    args = parser.parse_args()

    # [1]
    # Setting up simulator backend for blackbox repository
    # We use the NASBench201 blackbox for the training set `args.dataset`
    benchmark = nas201_benchmark(args.dataset)
    max_resource_attr = benchmark.max_resource_attr
    trial_backend = BlackboxRepositoryBackend(
        elapsed_time_attr=benchmark.elapsed_time_attr,
        max_resource_attr=max_resource_attr,
        blackbox_name=benchmark.blackbox_name,
        dataset=benchmark.dataset_name,
        surrogate=benchmark.surrogate,
        surrogate_kwargs=benchmark.surrogate_kwargs,
    )

    # [2]
    # Select configuration space for the benchmark. Here, we use the default
    # for the blackbox
    blackbox = trial_backend.blackbox
    # Common scheduler kwargs
    method_kwargs = dict(
        metric=benchmark.metric,
        mode=benchmark.mode,
        resource_attr=blackbox.fidelity_name(),
        random_seed=args.random_seed,
        max_resource_attr=max_resource_attr,
    )
    # Insert maximum resource level into configuration space. Doing so is
    # best practice and has advantages for pause-and-resume schedulers
    config_space = blackbox.configuration_space_with_max_resource_attr(
        max_resource_attr
    )

    scheduler = None
    if args.method in {"ASHA-STOP", "ASHA-PROM", "ASHA6-STOP"}:
        # [3]
        # Scheduler: Asynchronous Successive Halving (ASHA)
        # The 'stopping' variant stops trials which underperform compared to others
        # at certain resource levels (called rungs).
        # The 'promotion' variant pauses each trial at certain resource levels
        # (called rungs). Trials which outperform others at the same rung, are
        # promoted later on, to run to the next higher rung.
        # We configure this scheduler with random search: configurations for new
        # trials are drawn at random
        scheduler = ASHA(
            config_space,
            type="promotion" if args.method == "ASHA-PROM" else "stopping",
            brackets=6 if args.method == "ASHA6-STOP" else 1,
            **method_kwargs,
        )
    elif args.method in {"MOBSTER-JOINT", "MOBSTER-INDEP"}:
        # Scheduler: Asynchronous MOBSTER
        # We configure the scheduler with GP-based Bayesian optimization, using
        # the "gp_multitask" or the "gp_independent" surrogate model.
        search_options = None
        if args.method == "MOBSTER-INDEP":
            search_options = {"model": "gp_independent"}
        scheduler = MOBSTER(
            config_space,
            search_options=search_options,
            type="promotion",
            **method_kwargs,
        )
    elif args.method in {"HYPERTUNE-INDEP", "HYPERTUNE4-INDEP", "HYPERTUNE-JOINT"}:
        # Scheduler: Hyper-Tune
        # We configure the scheduler with GP-based Bayesian optimization, using
        # the "gp_multitask" or the "gp_independent" surrogate model.
        search_options = None
        if args.method == "HYPERTUNE-JOINT":
            search_options = {"model": "gp_multitask"}
        scheduler = HyperTune(
            config_space,
            search_options=search_options,
            type="promotion",
            brackets=4 if args.method == "HYPERTUNE4-INDEP" else 1,
            **method_kwargs,
        )
    elif args.method in {"SYNCHB", "SYNCSH"}:
        # Scheduler: Synchronous successive halving or Hyperband
        # We configure this scheduler with random search: configurations for new
        # trials are drawn at random
        scheduler = SyncHyperband(
            config_space,
            brackets=1 if args.method == "SYNCSH" else None,
            **method_kwargs,
        )
    elif args.method == "SYNCMOBSTER":
        # Scheduler: Synchronous MOBSTER
        # We configure this scheduler with GP-BO search. The default surrogate
        # model is "gp_independent": independent processes at each rung level,
        # which share a common ARD kernel, but separate mean functions and
        # covariance scales.
        scheduler = SyncMOBSTER(
            config_space,
            **method_kwargs,
        )
    elif args.method == "BOHB":
        # Scheduler: Synchronous BOHB
        # We configure this scheduler with KDE search, which is using the
        # "two-density" approximation of the EI acquisition function from
        # TPE (Bergstra & Bengio).
        scheduler = SyncBOHB(
            config_space,
            **method_kwargs,
        )
    elif args.method == "DEHB":
        # Scheduler: Differential Evolution Hyperband (DEHB)
        # We configure this scheduler with random search.
        scheduler = DEHB(
            config_space,
            **method_kwargs,
        )

    stop_criterion = StoppingCriterion(
        max_wallclock_time=benchmark.max_wallclock_time,
        max_num_evaluations=benchmark.max_num_evaluations,
    )

    # [4]
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=benchmark.n_workers,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
        tuner_name=args.experiment_tag,
        metadata={
            "seed": args.random_seed,
            "algorithm": args.method,
            "tag": args.experiment_tag,
            "benchmark": "nas201-" + args.dataset,
        },
    )

    tuner.run()
