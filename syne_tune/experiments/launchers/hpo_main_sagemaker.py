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
from typing import Optional

from syne_tune.experiments.baselines import MethodDefinitions
from syne_tune.experiments.launchers.hpo_main_common import (
    ExtraArgsType,
    MapMethodArgsType,
    ConfigDict,
    extra_metadata,
    DictStrKey,
    str2bool,
    config_from_argparse,
)
from syne_tune.experiments.launchers.hpo_main_local import (
    RealBenchmarkDefinitions,
    get_benchmark,
    create_objects_for_tuner,
    LOCAL_AND_SAGEMAKER_BACKEND_EXTRA_PARAMETERS,
)
from syne_tune.experiments.launchers.launch_remote_common import (
    sagemaker_estimator_args,
)
from syne_tune.experiments.launchers.utils import (
    get_master_random_seed,
)
from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    default_sagemaker_session,
)
from syne_tune.remote.estimators import sagemaker_estimator
from syne_tune.results_callback import ExtraResultsComposer
from syne_tune.tuner import Tuner

# SageMaker managed warm pools:
# https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html#train-warm-pools-resource-limits
# Maximum time a warm pool instance is kept alive, waiting to be associated with
# a new job. Setting this too large may lead to extra costs.
WARM_POOL_KEEP_ALIVE_PERIOD_IN_SECONDS = 10 * 60


SAGEMAKER_BACKEND_ONLY_EXTRA_PARAMETERS = [
    dict(
        name="max_failures",
        type=int,
        default=3,
        help="Number of trials which can fail without experiment being terminated",
    ),
    dict(
        name="warm_pool",
        type=str2bool,
        default=True,
        help=(
            "If 1, the SageMaker managed warm pools feature is used. "
            "This can be more expensive, but also reduces startup "
            "delays, leading to an experiment finishing in less time"
        ),
    ),
    dict(
        name="start_jobs_without_delay",
        type=str2bool,
        default=False,
        help=(
            "If 1, the tuner starts new trials immediately after "
            "sending existing ones a stop signal. This leads to more "
            "than n_workers instances being used during certain times, "
            "which can lead to quotas being exceeded, or the warm pool "
            "feature not working optimal."
        ),
    ),
]


SAGEMAKER_BACKEND_EXTRA_PARAMETERS = (
    LOCAL_AND_SAGEMAKER_BACKEND_EXTRA_PARAMETERS
    + SAGEMAKER_BACKEND_ONLY_EXTRA_PARAMETERS
)


def start_experiment_sagemaker_backend(
    configuration: ConfigDict,
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_results: Optional[ExtraResultsComposer] = None,
    map_method_args: Optional[MapMethodArgsType] = None,
    extra_tuning_job_metadata: Optional[DictStrKey] = None,
):
    """
    Runs experiment with SageMaker backend.

    ``map_method_args`` can be used to modify ``method_kwargs`` for constructing
    :class:`~syne_tune.experiments.baselines.MethodArguments`, depending on
    ``configuration`` and the method. This allows for extra flexibility to specify specific arguments for chosen methods
    Its signature is :code:`method_kwargs = map_method_args(configuration, method, method_kwargs)`,
    where ``method`` is the name of the baseline.

    :param configuration: ConfigDict with parameters of the experiment.
        Must contain all parameters from SAGEMAKER_BACKEND_EXTRA_PARAMETERS
    :param methods: Dictionary with method constructors.
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param extra_results: If given, this is used to append extra information to the
        results dataframe
    :param map_method_args: See above, optional
    :param extra_tuning_job_metadata: Metadata added to the tuner, can be used to manage results
    """
    configuration.check_if_all_paremeters_present(SAGEMAKER_BACKEND_EXTRA_PARAMETERS)
    configuration.expand_base_arguments(SAGEMAKER_BACKEND_EXTRA_PARAMETERS)

    experiment_tag = configuration.experiment_tag
    benchmark = get_benchmark(
        configuration, benchmark_definitions, sagemaker_backend=True
    )
    benchmark_name = configuration.benchmark
    master_random_seed = get_master_random_seed(configuration.random_seed)
    method_names = list(methods.keys())

    assert (
        len(method_names) == 1 and len(configuration.seeds) == 1
    ), "Can only launch single (method, seed). Use launch_remote to launch several combinations"
    method = method_names[0]
    seed = configuration.seeds[0]
    logging.getLogger().setLevel(logging.INFO)
    print(
        f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}"
        f"  max_wallclock_time = {benchmark.max_wallclock_time}, "
        f"  n_workers = {benchmark.n_workers}"
    )

    sm_args = sagemaker_estimator_args(
        entry_point=benchmark.script,
        experiment_tag="A",
        tuner_name="B",
        benchmark=benchmark,
    )
    del sm_args["checkpoint_s3_uri"]
    sm_args["sagemaker_session"] = default_sagemaker_session()
    if configuration.warm_pool:
        print(
            "--------------------------------------------------------------------------\n"
            "Using SageMaker managed warm pools in order to decrease start-up delays.\n"
            f"In order for this to work, you need to have at least {benchmark.n_workers} quotas of the type\n"
            f"   {benchmark.instance_type} for training warm pool usage\n"
            "--------------------------------------------------------------------------"
        )
        sm_args["keep_alive_period_in_seconds"] = WARM_POOL_KEEP_ALIVE_PERIOD_IN_SECONDS
    if configuration.instance_type is not None:
        sm_args["instance_type"] = configuration.instance_type
    trial_backend = SageMakerBackend(
        sm_estimator=sagemaker_estimator[benchmark.framework](**sm_args),
        # names of metrics to track. Each metric will be detected by Sagemaker if it is written in the
        # following form: "[RMSE]: 1.2", see in train_main_example how metrics are logged for an example
        delete_checkpoints=configuration.delete_checkpoints,
        metrics_names=[benchmark.metric],
    )

    tuner_kwargs = create_objects_for_tuner(
        configuration,
        methods=methods,
        method=method,
        benchmark=benchmark,
        master_random_seed=master_random_seed,
        seed=seed,
        verbose=True,
        extra_tuning_job_metadata=extra_tuning_job_metadata,
        map_method_args=map_method_args,
        extra_results=extra_results,
    )
    tuner = Tuner(
        trial_backend=trial_backend,
        **tuner_kwargs,
        sleep_time=5.0,
        max_failures=configuration.max_failures,
        start_jobs_without_delay=configuration.start_jobs_without_delay,
    )
    tuner.run()


def main(
    methods: MethodDefinitions,
    benchmark_definitions: RealBenchmarkDefinitions,
    extra_args: Optional[ExtraArgsType] = None,
    map_method_args: Optional[MapMethodArgsType] = None,
    extra_results: Optional[ExtraResultsComposer] = None,
):
    """
    Runs experiment with SageMaker backend.

    Command line arguments must specify a single benchmark, method, and seed,
    for example ``--method ASHA --num_seeds 5 --start_seed 4`` starts experiment
    with ``seed=4``, or ``--method ASHA --num_seeds 1`` starts experiment with
    ``seed=0``. Here, ``ASHA`` must be key in ``methods``.

    ``map_method_args`` can be used to modify ``method_kwargs`` for constructing
    :class:`~syne_tune.experiments.baselines.MethodArguments`, depending on
    ``configuration`` returned by :func:`parse_args` and the method. Its
    signature is
    :code:`method_kwargs = map_method_args(configuration, method, method_kwargs)`,
    where ``method`` is the name of the baseline. It is called just before the
    method is created.

    :param methods: Dictionary with method constructors
    :param benchmark_definitions: Definitions of benchmark; one is selected from
        command line arguments
    :param extra_args: Extra arguments for command line parser. Optional
    :param map_method_args: See above. Needed if ``extra_args`` is given
    :param extra_results: If given, this is used to append extra information to the
        results dataframe
    """
    configuration = config_from_argparse(extra_args, SAGEMAKER_BACKEND_EXTRA_PARAMETERS)
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

    start_experiment_sagemaker_backend(
        configuration,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        map_method_args=map_method_args,
        extra_results=extra_results,
        extra_tuning_job_metadata=None
        if extra_args is None
        else extra_metadata(configuration, extra_args),
    )
