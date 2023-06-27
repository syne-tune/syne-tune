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
from typing import Optional, List, Union, Dict, Any

import numpy as np
from tqdm import tqdm

from syne_tune.experiments.baselines import MethodArguments, MethodDefinitions
from syne_tune.experiments.benchmark_definitions.common import (
    SurrogateBenchmarkDefinition,
)
from syne_tune.experiments.launchers.hpo_main_common import (
    set_logging_level,
    get_metadata,
    ExtraArgsType,
    MapMethodArgsType,
    extra_metadata,
    ConfigDict,
    DictStrKey,
    str2bool,
    config_from_argparse,
)
from syne_tune.experiments.launchers.utils import (
    get_master_random_seed,
    effective_random_seed,
)
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
)
from syne_tune.results_callback import ExtraResultsComposer
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.util import sanitize_sagemaker_name


SIMULATED_BACKEND_EXTRA_PARAMETERS = [
    dict(
        name="benchmark",
        type=str,
        help="Benchmark to run from benchmark_definitions",
        default=None,
    ),
    dict(
        name="verbose",
        type=str2bool,
        default=False,
        help="Verbose log output?",
    ),
    dict(
        name="support_checkpointing",
        type=str2bool,
        default=1,
        help="If 0, trials are started from scratch when resumed",
    ),
    dict(
        name="fcnet_ordinal",
        type=str,
        choices=["none", "equal", "nn", "nn-log"],
        default="nn-log",
        help="Ordinal encoding for fcnet categorical HPs with numeric values. Use 'none' for categorical encoding",
    ),
    dict(
        name="restrict_configurations",
        type=str2bool,
        default=False,
        help="If 1, scheduler only suggests configs contained in tabulated benchmark",
    ),
]


BENCHMARK_KEY_EXTRA_PARAMETER = dict(
    name="benchmark_key",
    type=str,
    help="Key for benchmarks, needs to be specified if benchmarks definitions are nested.",
    default=None,
    required=True,
)


SurrogateBenchmarkDefinitions = Union[
    Dict[str, SurrogateBenchmarkDefinition],
    Dict[str, Dict[str, SurrogateBenchmarkDefinition]],
]


def is_dict_of_dict(benchmark_definitions: SurrogateBenchmarkDefinitions) -> bool:
    assert isinstance(benchmark_definitions, dict) and len(benchmark_definitions) > 0
    val = next(iter(benchmark_definitions.values()))
    return isinstance(val, dict)


def get_transfer_learning_evaluations(
    blackbox_name: str,
    test_task: str,
    datasets: Optional[List[str]],
    n_evals: Optional[int] = None,
) -> Dict[str, Any]:
    """
    :param blackbox_name: name of blackbox
    :param test_task: task where the performance would be tested, it is excluded from transfer-learning evaluations
    :param datasets: subset of datasets to consider, only evaluations from those datasets are provided to
    transfer-learning methods. If none, all datasets are used.
    :param n_evals: maximum number of evaluations to be returned
    :return:
    """
    task_to_evaluations = load_blackbox(blackbox_name)

    # todo retrieve right metric
    metric_index = 0
    transfer_learning_evaluations = {
        task: TransferLearningTaskEvaluations(
            configuration_space=bb.configuration_space,
            hyperparameters=bb.hyperparameters,
            objectives_evaluations=bb.objectives_evaluations[
                ..., metric_index : metric_index + 1
            ],
            objectives_names=[bb.objectives_names[metric_index]],
        )
        for task, bb in task_to_evaluations.items()
        if task != test_task and (datasets is None or task in datasets)
    }

    if n_evals is not None:
        # subsample n_evals / n_tasks of observations on each tasks
        def subsample(
            transfer_evaluations: TransferLearningTaskEvaluations, n: int
        ) -> TransferLearningTaskEvaluations:
            random_indices = np.random.permutation(
                len(transfer_evaluations.hyperparameters)
            )[:n]
            return TransferLearningTaskEvaluations(
                configuration_space=transfer_evaluations.configuration_space,
                hyperparameters=transfer_evaluations.hyperparameters.loc[
                    random_indices
                ].reset_index(drop=True),
                objectives_evaluations=transfer_evaluations.objectives_evaluations[
                    random_indices
                ],
                objectives_names=transfer_evaluations.objectives_names,
            )

        n = n_evals // len(transfer_learning_evaluations)
        transfer_learning_evaluations = {
            task: subsample(transfer_evaluations, n)
            for task, transfer_evaluations in transfer_learning_evaluations.items()
        }

    return transfer_learning_evaluations


def start_experiment_simulated_backend(
    configuration: ConfigDict,
    methods: MethodDefinitions,
    benchmark_definitions: SurrogateBenchmarkDefinitions,
    extra_results: Optional[ExtraResultsComposer] = None,
    map_method_args: Optional[MapMethodArgsType] = None,
    extra_tuning_job_metadata: Optional[DictStrKey] = None,
    use_transfer_learning: bool = False,
):
    """
    Runs sequence of experiments with simulator backend sequentially. The loop
    runs over methods selected from ``methods``, repetitions and benchmarks
    selected from ``benchmark_definitions``

    ``map_method_args`` can be used to modify ``method_kwargs`` for constructing
    :class:`~syne_tune.experiments.baselines.MethodArguments`, depending on
    ``configuration`` and the method. This allows for extra flexibility to specify specific arguments for chosen methods
    Its signature is :code:`method_kwargs = map_method_args(configuration, method, method_kwargs)`,
    where ``method`` is the name of the baseline.

    :param configuration: ConfigDict with parameters of the experiment.
        Must contain all parameters from LOCAL_LOCAL_SIMULATED_BENCHMARK_REQUIRED_PARAMETERS
    :param methods: Dictionary with method constructors.
    :param benchmark_definitions: Definitions of benchmarks; one is selected from
        command line arguments
    :param extra_results: If given, this is used to append extra information to the
        results dataframe
    :param map_method_args: See above, optional
    :param extra_tuning_job_metadata: Metadata added to the tuner, can be used to manage results
    :param use_transfer_learning: If True, we use transfer tuning. Defaults to
        False
    """
    simulated_backend_extra_parameters = SIMULATED_BACKEND_EXTRA_PARAMETERS.copy()
    nested_dict = is_dict_of_dict(benchmark_definitions)
    if nested_dict:
        simulated_backend_extra_parameters.append(BENCHMARK_KEY_EXTRA_PARAMETER)
    configuration.check_if_all_paremeters_present(simulated_backend_extra_parameters)
    configuration.expand_base_arguments(simulated_backend_extra_parameters)

    if configuration.benchmark is not None:
        benchmark_names = [configuration.benchmark]
    else:
        if nested_dict:
            # If ``parse_args`` is called from ``launch_remote``, ``benchmark_key`` is
            # not set. In this case, ``benchmark_names`` is not needed
            k = configuration.benchmark_key
            if k is None:
                bm_dict = dict()
            else:
                bm_dict = benchmark_definitions.get(k)
                assert (
                    bm_dict is not None
                ), f"{k} (value of --benchmark_key) is not among keys of benchmark_definition [{list(benchmark_definitions.keys())}]"
        else:
            bm_dict = benchmark_definitions
        benchmark_names = list(bm_dict.keys())

    method_names = list(methods.keys())
    experiment_tag = configuration.experiment_tag
    master_random_seed = get_master_random_seed(configuration.random_seed)
    if is_dict_of_dict(benchmark_definitions):
        assert (
            configuration.benchmark_key is not None
        ), "Use --benchmark_key if benchmark_definitions is a nested dictionary"
        benchmark_definitions = benchmark_definitions[configuration.benchmark_key]
    set_logging_level(configuration)

    combinations = list(
        itertools.product(method_names, configuration.seeds, benchmark_names)
    )
    print(combinations)
    do_scale = (
        configuration.scale_max_wallclock_time
        and configuration.n_workers is not None
        and configuration.max_wallclock_time is None
    )
    for method, seed, benchmark_name in tqdm(combinations):
        random_seed = effective_random_seed(master_random_seed, seed)
        np.random.seed(random_seed)
        benchmark = benchmark_definitions[benchmark_name]
        default_n_workers = benchmark.n_workers
        if configuration.n_workers is not None:
            benchmark.n_workers = configuration.n_workers
        if configuration.max_wallclock_time is not None:
            benchmark.max_wallclock_time = configuration.max_wallclock_time
        elif do_scale and configuration.n_workers < default_n_workers:
            # Scale ``max_wallclock_time``
            factor = default_n_workers / configuration.n_workers
            bm_mwt = benchmark.max_wallclock_time
            benchmark.max_wallclock_time = int(bm_mwt * factor)
            print(
                f"Scaling max_wallclock_time: {benchmark.max_wallclock_time} (from {bm_mwt})"
            )
        print(
            f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}"
            f"  max_wallclock_time = {benchmark.max_wallclock_time}, "
            f"  n_workers = {benchmark.n_workers}"
        )

        max_resource_attr = benchmark.max_resource_attr
        if max_resource_attr is None:
            max_resource_attr = "my_max_resource_attr"
        if configuration.restrict_configurations:
            # Don't need surrogate in this case
            kwargs = dict()
        else:
            kwargs = dict(
                surrogate=benchmark.surrogate,
                surrogate_kwargs=benchmark.surrogate_kwargs,
                add_surrogate_kwargs=benchmark.add_surrogate_kwargs,
            )
        trial_backend = BlackboxRepositoryBackend(
            blackbox_name=benchmark.blackbox_name,
            elapsed_time_attr=benchmark.elapsed_time_attr,
            max_resource_attr=max_resource_attr,
            support_checkpointing=configuration.support_checkpointing,
            dataset=benchmark.dataset_name,
            **kwargs,
        )

        blackbox = trial_backend.blackbox
        resource_attr = blackbox.fidelity_name()
        config_space = blackbox.configuration_space_with_max_resource_attr(
            max_resource_attr
        )
        method_kwargs = dict(
            config_space=config_space,
            metric=benchmark.metric,
            mode=benchmark.mode,
            random_seed=random_seed,
            resource_attr=resource_attr,
            max_resource_attr=max_resource_attr,
            use_surrogates="lcbench" in benchmark_name,
            fcnet_ordinal=configuration.fcnet_ordinal,
            scheduler_kwargs=dict(
                points_to_evaluate=benchmark.points_to_evaluate,
            ),
        )
        if use_transfer_learning:
            method_kwargs["transfer_learning_evaluations"] = (
                get_transfer_learning_evaluations(
                    blackbox_name=benchmark.blackbox_name,
                    test_task=benchmark.dataset_name,
                    datasets=benchmark.datasets,
                ),
            )
        search_options = dict(debug_log=configuration.verbose)
        if configuration.restrict_configurations:
            search_options["restrict_configurations"] = blackbox.all_configurations()
        if configuration.max_size_data_for_model is not None:
            search_options[
                "max_size_data_for_model"
            ] = configuration.max_size_data_for_model
        method_kwargs["scheduler_kwargs"]["search_options"] = search_options
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
            experiment_tag=experiment_tag,
            benchmark_name=benchmark_name,
            random_seed=master_random_seed,
            max_size_data_for_model=configuration.max_size_data_for_model,
            extra_metadata=extra_tuning_job_metadata,
        )
        metadata["fcnet_ordinal"] = configuration.fcnet_ordinal
        if benchmark.add_surrogate_kwargs is not None:
            metadata["predict_curves"] = int(
                benchmark.add_surrogate_kwargs["predict_curves"]
            )
        tuner_name = experiment_tag
        if configuration.use_long_tuner_name_prefix:
            tuner_name += f"-{sanitize_sagemaker_name(benchmark_name)}-{seed}"
        callbacks = [SimulatorCallback(extra_results_composer=extra_results)]
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=benchmark.n_workers,
            sleep_time=0,
            callbacks=callbacks,
            results_update_interval=600,
            print_update_interval=600,
            tuner_name=tuner_name,
            metadata=metadata,
            save_tuner=configuration.save_tuner,
        )
        tuner.run()


def main(
    methods: MethodDefinitions,
    benchmark_definitions: SurrogateBenchmarkDefinitions,
    extra_args: Optional[ExtraArgsType] = None,
    map_method_args: Optional[MapMethodArgsType] = None,
    extra_results: Optional[ExtraResultsComposer] = None,
    use_transfer_learning: bool = False,
):
    """
    Runs sequence of experiments with simulator backend sequentially. The loop
    runs over methods selected from ``methods``, repetitions and benchmarks
    selected from ``benchmark_definitions``, with the range being controlled by
    command line arguments.

    ``map_method_args`` can be used to modify ``method_kwargs`` for constructing
    :class:`~syne_tune.experiments.baselines.MethodArguments`, depending on
    ``configuration`` returned by :func:`parse_args` and the method. Its
    signature is
    :code:`method_kwargs = map_method_args(configuration, method, method_kwargs)`,
    where ``method`` is the name of the baseline. It is called just before the
    method is created.

    :param methods: Dictionary with method constructors
    :param benchmark_definitions: Definitions of benchmarks
    :param extra_args: Extra arguments for command line parser. Optional
    :param map_method_args: See above. Needed if ``extra_args`` given
    :param extra_results: If given, this is used to append extra information to the
        results dataframe
    :param use_transfer_learning: If True, we use transfer tuning. Defaults to
        False
    """
    simulated_backend_extra_parameters = SIMULATED_BACKEND_EXTRA_PARAMETERS.copy()
    if is_dict_of_dict(benchmark_definitions):
        simulated_backend_extra_parameters.append(BENCHMARK_KEY_EXTRA_PARAMETER)
    configuration = config_from_argparse(extra_args, simulated_backend_extra_parameters)

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

    start_experiment_simulated_backend(
        configuration,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        map_method_args=map_method_args,
        extra_results=extra_results,
        extra_tuning_job_metadata=None
        if extra_args is None
        else extra_metadata(configuration, extra_args),
        use_transfer_learning=use_transfer_learning,
    )
