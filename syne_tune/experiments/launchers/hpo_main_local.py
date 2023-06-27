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
import itertools
from typing import Optional, Callable, Dict, Any
import numpy as np
from tqdm import tqdm
import logging

from syne_tune.experiments.baselines import MethodArguments, MethodDefinitions
from syne_tune.experiments.benchmark_definitions.common import RealBenchmarkDefinition
from syne_tune.experiments.launchers.hpo_main_common import (
    set_logging_level,
    get_metadata,
    ExtraArgsType,
    MapMethodArgsType,
    ConfigDict,
    DictStrKey,
    extra_metadata,
    str2bool,
    config_from_argparse,
)
from syne_tune.experiments.launchers.utils import (
    get_master_random_seed,
    effective_random_seed,
)
from syne_tune.backend import LocalBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    backend_path_not_synced_to_s3,
)
from syne_tune.remote.remote_metrics_callback import RemoteTuningMetricsCallback
from syne_tune.results_callback import StoreResultsCallback, ExtraResultsComposer
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.util import sanitize_sagemaker_name

logger = logging.getLogger(__name__)


RealBenchmarkDefinitions = Callable[..., Dict[str, RealBenchmarkDefinition]]


LOCAL_AND_SAGEMAKER_BACKEND_EXTRA_PARAMETERS = [
    dict(
        name="benchmark",
        type=str,
        help="Benchmark to run from benchmark_definitions",
        default=None,
    ),
    dict(
        name="instance_type",
        type=str,
        default=None,
        help="AWS SageMaker instance type",
    ),
    dict(
        name="delete_checkpoints",
        type=str2bool,
        default=False,
        help="Remove checkpoints of trials once no longer needed?",
    ),
    dict(
        name="remote_tuning_metrics",
        type=str2bool,
        default=True,
        help="Remote tuning publishes metrics to Sagemaker console?",
    ),
]


LOCAL_BACKEND_ONLY_EXTRA_PARAMETERS = [
    dict(
        name="verbose",
        type=str2bool,
        default=False,
        help="Verbose log output?",
    ),
    dict(
        name="num_gpus_per_trial",
        type=int,
        default=1,
        help="Number of GPUs to allocate to each trial",
    ),
]


LOCAL_BACKEND_EXTRA_PARAMETERS = (
    LOCAL_AND_SAGEMAKER_BACKEND_EXTRA_PARAMETERS + LOCAL_BACKEND_ONLY_EXTRA_PARAMETERS
)


def get_benchmark(
    configuration: ConfigDict,
    benchmark_definitions: RealBenchmarkDefinitions,
    **benchmark_kwargs,
) -> RealBenchmarkDefinition:
    """
    If ``configuration.benchmark`` is ``None`` and ``benchmark_definitions`` maps
    to a single benchmark, ``configuration.benchmark`` is set to its key.
    """
    if configuration.n_workers is not None:
        benchmark_kwargs["n_workers"] = configuration.n_workers
    if configuration.max_wallclock_time is not None:
        benchmark_kwargs["max_wallclock_time"] = configuration.max_wallclock_time
    if configuration.instance_type is not None:
        benchmark_kwargs["instance_type"] = configuration.instance_type
    benchmarks_dict = benchmark_definitions(**benchmark_kwargs)
    if configuration.benchmark is None:
        assert (
            len(benchmarks_dict) == 1
        ), f"--benchmark must be given if benchmark_definitions has more than one entry"
        configuration.benchmark = next(iter(benchmarks_dict.keys()))
    benchmark = benchmarks_dict[configuration.benchmark]
    do_scale = (
        configuration.scale_max_wallclock_time
        and configuration.n_workers is not None
        and configuration.max_wallclock_time is None
    )
    if do_scale:
        benchmark_default = benchmark_definitions(**benchmark_kwargs)[
            configuration.benchmark
        ]
        default_n_workers = benchmark_default.n_workers
    else:
        default_n_workers = None
    if do_scale and configuration.n_workers < default_n_workers:
        # Scale ``max_wallclock_time``
        factor = default_n_workers / configuration.n_workers
        bm_mwt = benchmark.max_wallclock_time
        benchmark.max_wallclock_time = int(bm_mwt * factor)
        print(
            f"Scaling max_wallclock_time: {benchmark.max_wallclock_time} (from {bm_mwt})"
        )
    return benchmark


def create_objects_for_tuner(
    configuration: ConfigDict,
    methods: MethodDefinitions,
    method: str,
    benchmark: RealBenchmarkDefinition,
    master_random_seed: int,
    seed: int,
    verbose: bool,
    extra_tuning_job_metadata: Optional[DictStrKey] = None,
    map_method_args: Optional[MapMethodArgsType] = None,
    extra_results: Optional[ExtraResultsComposer] = None,
    num_gpus_per_trial: int = 1,
) -> Dict[str, Any]:
    method_kwargs = dict(
        config_space=benchmark.config_space,
        metric=benchmark.metric,
        mode=benchmark.mode,
        random_seed=effective_random_seed(master_random_seed, seed),
        resource_attr=benchmark.resource_attr,
        max_resource_attr=benchmark.max_resource_attr,
        num_gpus_per_trial=num_gpus_per_trial,
        scheduler_kwargs=dict(
            points_to_evaluate=benchmark.points_to_evaluate,
            search_options=dict(debug_log=verbose),
        ),
    )
    if configuration.max_size_data_for_model is not None:
        method_kwargs["scheduler_kwargs"]["search_options"][
            "max_size_data_for_model"
        ] = configuration.max_size_data_for_model
    if map_method_args is not None:
        method_kwargs = map_method_args(configuration, method, method_kwargs)
    scheduler = methods[method](MethodArguments(**method_kwargs))

    stop_criterion = StoppingCriterion(
        max_wallclock_time=benchmark.max_wallclock_time,
        max_num_evaluations=benchmark.max_num_evaluations,
    )
    metadata = get_metadata(
        seed=seed,
        method=method,
        experiment_tag=configuration.experiment_tag,
        benchmark_name=configuration.benchmark,
        random_seed=master_random_seed,
        max_size_data_for_model=configuration.max_size_data_for_model,
        benchmark=benchmark,
        extra_metadata=extra_tuning_job_metadata,
    )
    metadata["instance_type"] = benchmark.instance_type
    metadata["num_gpus_per_trial"] = num_gpus_per_trial
    tuner_name = configuration.experiment_tag
    if configuration.use_long_tuner_name_prefix:
        tuner_name += f"-{sanitize_sagemaker_name(configuration.benchmark)}-{seed}"
    tuner_kwargs = dict(
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=benchmark.n_workers,
        tuner_name=tuner_name,
        metadata=metadata,
        save_tuner=configuration.save_tuner,
    )
    callbacks = [StoreResultsCallback(extra_results_composer=extra_results)]
    if configuration.remote_tuning_metrics:
        # Use callback to report tuning metrics
        callbacks.append(
            RemoteTuningMetricsCallback(
                metric=benchmark.metric,
                mode=benchmark.mode,
                config_space=benchmark.config_space,
                resource_attr=benchmark.resource_attr,
            )
        )
    tuner_kwargs["callbacks"] = callbacks
    return tuner_kwargs


def start_experiment_local_backend(
    configuration: ConfigDict,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_results: Optional[ExtraResultsComposer] = None,
    map_method_args: Optional[MapMethodArgsType] = None,
    extra_tuning_job_metadata: Optional[DictStrKey] = None,
):
    """
    Runs sequence of experiments with local backend sequentially.
    The loop runs over methods selected from ``methods`` and repetitions,

    ``map_method_args`` can be used to modify ``method_kwargs`` for constructing
    :class:`~syne_tune.experiments.baselines.MethodArguments`, depending on
    ``configuration`` and the method. This allows for extra flexibility to specify specific arguments for chosen methods
    Its signature is :code:`method_kwargs = map_method_args(configuration, method, method_kwargs)`,
    where ``method`` is the name of the baseline.

    .. note::
       When this is launched remotely as entry point of a SageMaker training
       job (command line ``--launched_remotely 1``), the backend is configured
       to write logs and checkpoints to a directory which is not synced to S3.
       This is different to the tuner path, which is "/opt/ml/checkpoints", so
       that tuning results are synced to S3. Syncing checkpoints to S3 is not
       recommended (it is slow and can lead to failures, since several worker
       processes write to the same synced directory).

    :param configuration: ConfigDict with parameters of the experiment.
        Must contain all parameters from LOCAL_BACKEND_EXTRA_PARAMETERS
    :param methods: Dictionary with method constructors.
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param extra_results: If given, this is used to append extra information to the
        results dataframe
    :param map_method_args: See above, optional
    :param extra_tuning_job_metadata: Metadata added to the tuner, can be used to manage results
    """
    configuration.check_if_all_paremeters_present(LOCAL_BACKEND_EXTRA_PARAMETERS)
    configuration.expand_base_arguments(LOCAL_BACKEND_EXTRA_PARAMETERS)

    experiment_tag = configuration.experiment_tag
    benchmark = get_benchmark(configuration, benchmark_definitions)
    benchmark_name = configuration.benchmark
    master_random_seed = get_master_random_seed(configuration.random_seed)
    set_logging_level(configuration)

    combinations = list(itertools.product(list(methods.keys()), configuration.seeds))
    print(combinations)
    for method, seed in tqdm(combinations):
        random_seed = effective_random_seed(master_random_seed, seed)
        np.random.seed(random_seed)
        print(
            f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}\n"
            f"  max_wallclock_time = {benchmark.max_wallclock_time}, "
            f"  n_workers = {benchmark.n_workers}"
        )
        trial_backend = LocalBackend(
            entry_point=str(benchmark.script),
            delete_checkpoints=configuration.delete_checkpoints,
            num_gpus_per_trial=configuration.num_gpus_per_trial,
        )

        tuner_kwargs = create_objects_for_tuner(
            configuration,
            methods=methods,
            method=method,
            benchmark=benchmark,
            master_random_seed=master_random_seed,
            seed=seed,
            verbose=configuration.verbose,
            extra_tuning_job_metadata=extra_tuning_job_metadata,
            map_method_args=map_method_args,
            extra_results=extra_results,
            num_gpus_per_trial=configuration.num_gpus_per_trial,
        )
        # If this experiments runs remotely as a SageMaker training job, logs and
        # checkpoints are written to a different directory than tuning results, so
        # the former are not synced to S3.
        if configuration.launched_remotely:
            tuner_kwargs["trial_backend_path"] = backend_path_not_synced_to_s3()
        tuner = Tuner(
            trial_backend=trial_backend,
            **tuner_kwargs,
        )

        tuner.run()  # Run the experiment


def main(
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[ExtraArgsType] = None,
    map_method_args: Optional[MapMethodArgsType] = None,
    extra_results: Optional[ExtraResultsComposer] = None,
):
    """
    Runs sequence of experiments with local backend sequentially. The loop runs
    over methods selected from ``methods`` and repetitions, both controlled by
    command line arguments.

    ``map_method_args`` can be used to modify ``method_kwargs`` for constructing
    :class:`~syne_tune.experiments.baselines.MethodArguments`, depending on
    ``configuration`` returned by :func:`parse_args` and the method. Its
    signature is
    :code:`method_kwargs = map_method_args(configuration, method, method_kwargs)`,
    where ``method`` is the name of the baseline. It is called just before the
    method is created.

    :param methods: Dictionary with method constructors
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param extra_args: Extra arguments for command line parser. Optional
    :param map_method_args: See above, optional
    :param extra_results: If given, this is used to append extra information to the
        results dataframe
    """
    configuration = config_from_argparse(extra_args, LOCAL_BACKEND_EXTRA_PARAMETERS)
    method_names = (
        [configuration.method]
        if configuration.method is not None
        else list(methods.keys())
    )
    methods = {mname: methods[mname] for mname in method_names}
    if extra_args is not None:
        assert (
            map_method_args is not None
        ), "map_method_args must be specified if extra_args is used"

    start_experiment_local_backend(
        configuration,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        map_method_args=map_method_args,
        extra_results=extra_results,
        extra_tuning_job_metadata=None
        if extra_args is None
        else extra_metadata(configuration, extra_args),
    )
