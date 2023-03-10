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
from typing import Optional, List, Callable, Dict, Any
import numpy as np
import itertools
from tqdm import tqdm
import logging

from syne_tune.backend import LocalBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    set_backend_path_not_synced_to_s3,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.util import sanitize_sagemaker_name
from benchmarking.commons.baselines import MethodArguments, MethodDefinitions
from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.commons.hpo_main_common import (
    parse_args as _parse_args,
    set_logging_level,
    get_metadata,
    ExtraArgsType,
    MapExtraArgsType,
    PostProcessingType,
    extra_metadata,
)
from benchmarking.commons.utils import get_master_random_seed, effective_random_seed

logger = logging.getLogger(__name__)


RealBenchmarkDefinitions = Callable[..., Dict[str, RealBenchmarkDefinition]]


def get_benchmark(
    args, benchmark_definitions: RealBenchmarkDefinitions, **benchmark_kwargs
) -> RealBenchmarkDefinition:
    do_scale = (
        args.scale_max_wallclock_time
        and args.n_workers is not None
        and args.max_wallclock_time is None
    )
    if do_scale:
        benchmark_default = benchmark_definitions(**benchmark_kwargs)[args.benchmark]
        default_n_workers = benchmark_default.n_workers
    else:
        default_n_workers = None
    if args.n_workers is not None:
        benchmark_kwargs["n_workers"] = args.n_workers
    if args.max_wallclock_time is not None:
        benchmark_kwargs["max_wallclock_time"] = args.max_wallclock_time
    if args.instance_type is not None:
        benchmark_kwargs["instance_type"] = args.instance_type
    benchmark = benchmark_definitions(**benchmark_kwargs)[args.benchmark]
    if do_scale and args.n_workers < default_n_workers:
        # Scale ``max_wallclock_time``
        factor = default_n_workers / args.n_workers
        bm_mwt = benchmark.max_wallclock_time
        benchmark.max_wallclock_time = int(bm_mwt * factor)
        print(
            f"Scaling max_wallclock_time: {benchmark.max_wallclock_time} (from {bm_mwt})"
        )
    return benchmark


def parse_args(
    methods: Dict[str, Any], extra_args: Optional[ExtraArgsType] = None
) -> (Any, List[str], List[int]):
    """Parse command line arguments for local backend experiments.

    :param methods: If ``--method`` is not given, then ``method_names`` are the
        keys of this dictionary
    :param extra_args: List of dictionaries, containing additional arguments
        to be passed. Must contain ``name`` for argument name (without leading
        ``"--"``), and other kwargs to ``parser.add_argument``. Optional
    :return: ``(args, method_names, seeds)``, where ``args`` is result of
        ``parser.parse_args()``, ``method_names`` see ``methods``, and
        ``seeds`` are list of seeds specified by ``--num_seeds`` and ``--start_seed``
    """
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
                name="verbose",
                type=int,
                default=0,
                help="Verbose log output?",
            ),
            dict(
                name="instance_type",
                type=str,
                help="AWS SageMaker instance type",
            ),
        ]
    )
    args, method_names, seeds = _parse_args(methods, extra_args)
    args.verbose = bool(args.verbose)
    return args, method_names, seeds


def create_objects_for_tuner(
    args,
    methods: MethodDefinitions,
    extra_args: Optional[ExtraArgsType],
    map_extra_args: Optional[MapExtraArgsType],
    method: str,
    benchmark: RealBenchmarkDefinition,
    master_random_seed: int,
    seed: int,
    verbose: bool,
) -> Dict[str, Any]:
    method_kwargs = {"max_resource_attr": benchmark.max_resource_attr}
    if args.max_size_data_for_model is not None:
        method_kwargs["scheduler_kwargs"] = {
            "search_options": {"max_size_data_for_model": args.max_size_data_for_model},
        }
    if extra_args is not None:
        assert map_extra_args is not None
        method_kwargs = map_extra_args(args, method, method_kwargs)
    method_kwargs.update(
        dict(
            config_space=benchmark.config_space,
            metric=benchmark.metric,
            mode=benchmark.mode,
            random_seed=effective_random_seed(master_random_seed, seed),
            resource_attr=benchmark.resource_attr,
            verbose=verbose,
        )
    )
    scheduler = methods[method](MethodArguments(**method_kwargs))

    stop_criterion = StoppingCriterion(
        max_wallclock_time=benchmark.max_wallclock_time,
        max_num_evaluations=benchmark.max_num_evaluations,
    )
    metadata = get_metadata(
        seed=seed,
        method=method,
        experiment_tag=args.experiment_tag,
        benchmark_name=args.benchmark,
        random_seed=master_random_seed,
        max_size_data_for_model=args.max_size_data_for_model,
        benchmark=benchmark,
        extra_args=None if extra_args is None else extra_metadata(args, extra_args),
    )
    tuner_name = args.experiment_tag
    if args.use_long_tuner_name_prefix:
        tuner_name += f"-{sanitize_sagemaker_name(args.benchmark)}-{seed}"
    return dict(
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=benchmark.n_workers,
        tuner_name=tuner_name,
        metadata=metadata,
        save_tuner=args.save_tuner,
    )


def main(
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[ExtraArgsType] = None,
    map_extra_args: Optional[MapExtraArgsType] = None,
    post_processing: Optional[PostProcessingType] = None,
):
    """
    Runs sequence of experiments with local backend sequentially. The loop runs
    over methods selected from ``methods`` and repetitions, both controlled by
    command line arguments.

    ``map_extra_args`` can be used to modify ``method_kwargs`` for constructing
    :class:`~benchmarking.commons.baselines.MethodArguments`, depending on
    ``args`` returned by :func:`parse_args` and the method. Its signature is
    :code:`method_kwargs = map_extra_args(args, method, method_kwargs)`, where
    ``method`` is the name of the baseline.

    .. note::
       When this is launched remotely as entry point of a SageMaker training
       job (command line ``--launched_remotely 1``), the backend is configured
       to write logs and checkpoints to a directory which is not synced to S3.
       This is different to the tuner path, which is "/opt/ml/checkpoints", so
       that tuning results are synced to S3. Syncing checkpoints to S3 is not
       recommended (it is slow and can lead to failures, since several worker
       processes write to the same synced directory).

    :param methods: Dictionary with method constructors
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param extra_args: Extra arguments for command line parser. Optional
    :param map_extra_args: See above, optional
    :param post_processing: Called after tuning has finished, passing the tuner
        as argument. Can be used for postprocessing, such as output or storage
        of extra information
    """
    args, method_names, seeds = parse_args(methods, extra_args)
    experiment_tag = args.experiment_tag
    benchmark_name = args.benchmark
    master_random_seed = get_master_random_seed(args.random_seed)
    set_logging_level(args)
    benchmark = get_benchmark(args, benchmark_definitions)

    combinations = list(itertools.product(method_names, seeds))
    print(combinations)
    for method, seed in tqdm(combinations):
        random_seed = effective_random_seed(master_random_seed, seed)
        np.random.seed(random_seed)
        print(
            f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}"
        )
        trial_backend = LocalBackend(entry_point=str(benchmark.script))

        tuner_kwargs = create_objects_for_tuner(
            args,
            methods=methods,
            extra_args=extra_args,
            map_extra_args=map_extra_args,
            method=method,
            benchmark=benchmark,
            master_random_seed=master_random_seed,
            seed=seed,
            verbose=args.verbose,
        )
        tuner = Tuner(
            trial_backend=trial_backend,
            **tuner_kwargs,
        )
        # If this experiments runs remotely as a SageMaker training job, logs and
        # checkpoints are written to a different directory than tuning results, so
        # the former are not synced to S3.
        # Note: This has to be done after ``tuner`` is created, because this calls
        # ``trial_backend.set_path`` as well.
        if args.launched_remotely:
            set_backend_path_not_synced_to_s3(trial_backend)

        tuner.run()  # Run the experiment
        if post_processing is not None:
            post_processing(tuner)
