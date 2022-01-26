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
from pathlib import Path
import itertools
import copy
import numpy as np

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.backend.sagemaker_backend.sagemaker_backend import \
    SagemakerBackend
from syne_tune.backend.simulator_backend.simulator_backend import \
    SimulatorBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.remote.remote_launcher import RemoteLauncher
from syne_tune.util import s3_experiment_path

from benchmarking.cli.estimator_factory import sagemaker_estimator_factory
from benchmarking.cli.launch_utils import parse_args
from benchmarking.cli.benchmark_factory import benchmark_factory
from benchmarking.cli.scheduler_factory import scheduler_factory
from benchmarking.utils.dict_get import dict_get

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    """
    Example for calling the CLI:
    
    python benchmarks/launch_hpo.py --scheduler hyperband_stopping \
        --searcher bayesopt --benchmark_name mlp_fashionmnist \
        --scheduler_timeout 120 --local_tuner
    
    See also `docs/command_line.md`.

    This is launching a single experiment. You can launch several experiments
    by passing list values for certain arguments (most arguments support list
    values). If you do this, the combinatorial product of all combinations
    is iterated over.
    
    Example:
    
    python benchmarks/launch_hpo.py --scheduler fifo hyperband_stopping \
        --searcher bayesopt random \
        --benchmark_name mlp_fashionmnist \
        --scheduler_timeout 120 --local_tuner

    is launching 4 experiments, with (scheduler, searcher) taking on values
    (fifo, bayesopt), (hyperband_stopping, bayesopt), (fifo, random),
    (hyperband_stopping, random).
    
    If you use --local_tuner, these experiments are run in sequence. If you drop
    --local_tuner, experiments are launched remotely as SageMaker jobs, so
    they can all run in parallel.

    Iterating over all combinations is not always what you want. You can use
    --argument_groups in order to group list value arguments together. The list
    values of arguments in the same group are iterated over jointly (so zip
    instead of product).
    
    Example:
    
    python benchmarks/launch_hpo.py \
        --scheduler hyperband_stopping fifo \
        --searcher bayesopt random \
        --max_resource_level 9 27 81 \
        --benchmark_name mlp_fashionmnist \
        --scheduler_timeout 120 --local_tuner \
        --argument_groups "scheduler searcher"

    is launching 2 * 3 = 6 experiments, with (scheduler, searcher,
    max_resource_level) taking on values
    (hyperband_stopping, bayesopt, 9), (fifo, random, 9),
    (hyperband_stopping, bayesopt, 27), (fifo, random, 27),
    (hyperband_stopping, bayesopt, 81), (fifo, random, 81).
    
    Several groups can be formed.
    
    Example:
    
    python benchmarks/launch_hpo.py \
        --scheduler hyperband_stopping fifo \
        --searcher bayesopt random \
        --benchmark_name mlp_fashionmnist \
        --max_resource_level 27 81 \
        --scheduler_timeout 120 240 --local_tuner \
        --argument_groups "scheduler searcher|max_resource_level scheduler_timeout"
    
    is launching 2 * 2 = 4 experiments, with (scheduler, searcher,
    max_resource_level, scheduler_timeout) taking on values
    (hyperband_stopping, bayesopt, 27, 120), (fifo, random, 27, 120),
    (hyperband_stopping, bayesopt, 81, 240), (fifo, random, 81, 240).
    
    A special argument with list values is --run_id, its values must be
    distinct nonnegative integers. Here, "--num_runs 5" is short for
    "--run_id 0 1 2 3 4". If --run_id is given, --num_runs is ignored.
    If neither of the two is given, the default is run_id = 0.

    When multiple experiments are launched in this way, you can use the
    --skip_initial_experiments argument in order to skip this number of initial
    experiments before launching the remaining ones. This is useful if a
    previous call failed to launch all intended experiments (e.g., because an
    AWS instance limit was reached). If the initial K experiments were in
    fact launched, a subsequent call with --skip_initial_experiments K will
    launch only the remaining ones.
    
    Note that --benchmark_name does not support list values. This is because
    we also support benchmark-specific command line arguments.

    """
    orig_params = parse_args(allow_lists_as_values=True)
    if orig_params['debug_log_level'] and orig_params['local_tuner']:
        # For remote tuning, 'debug_log_level' concerns the remote tuning
        # job, not the local one here (where logging.DEBUG just pollutes
        # the output)
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.getLogger().setLevel(log_level)

    # Basic checks not done in `parse_args`
    run_id = orig_params.get('run_id')
    if run_id is not None:
        if not isinstance(run_id, list):
            run_id = [run_id]
        else:
            assert len(set(run_id)) == len(run_id), \
                f"run_id = {run_id} contains duplicate entries"
        for id in run_id:
            assert id >= 0, f"run_id contains negative entry {id}"
    else:
        # Alternative to specify `run_id`
        num_runs = orig_params.get('num_runs')
        if num_runs is None:
            run_id = [0]
        else:
            assert num_runs >=1, f"num_runs = {num_runs} must be positive"
            run_id = list(range(num_runs))
            del orig_params['num_runs']
        orig_params['run_id'] = run_id

    # Master random seed is offset plus run_id, modulo 2 ** 32. The offset
    # is drawn at random and displayed if not specified.
    random_seed_offset = orig_params.get('random_seed_offset')
    if random_seed_offset is None:
        random_seed_offset = np.random.randint(0, 2 ** 32)
        orig_params['random_seed_offset'] = random_seed_offset
    logger.info(f"Using random_seed_offset = {random_seed_offset}")

    # Split into params with list values and normal values
    params_listvals = dict()
    params_nolistvals = dict()
    keys_with_list = []
    for k, v in orig_params.items():
        if isinstance(v, list):
            params_listvals[k] = v
            if len(v) > 1:
                keys_with_list.append(k)
        else:
            params_nolistvals[k] = v
    argument_groups = orig_params.get('argument_groups')
    # Group list arguments together
    if argument_groups is not None:
        groups = [x.strip() for x in argument_groups.split('|')]
        for group in groups:
            list_lists = []
            keys = group.split()
            # Singleton groups can be ignored
            if len(keys) > 1:
                for k in keys:
                    v = params_listvals.get(k)
                    assert v is not None, \
                        f"{k} in argument_groups group {group} is not a list argument"
                    if list_lists:
                        assert len(v) == len(list_lists[0]), \
                            f"Lists value in group {group} must all have " +\
                            "the same length"
                    list_lists.append(v)
                    del params_listvals[k]
                group_key = '|'.join(keys)
                params_listvals[group_key] = list(zip(*list_lists))
    num_experiments = 1
    keys = []
    list_values = []
    for k, v in params_listvals.items():
        keys.append(k)
        list_values.append(v)
        num_experiments *= len(v)

    skip_initial_experiments = orig_params['skip_initial_experiments']
    assert skip_initial_experiments >= 0, \
        "--skip_initial_experiments must be nonnegative"
    if num_experiments > 1:
        msg = f"The following arguments have list values: {keys_with_list}\n" +\
              f"Total number of experiments: {num_experiments}"
        if skip_initial_experiments > 0:
            msg += f"\nSkipping {skip_initial_experiments}, launching " +\
                   f"{num_experiments - skip_initial_experiments}"
        logger.info(msg)

    # Loop over all combinations
    experiment_name = dict_get(orig_params, 'experiment_name', 'stune')
    s3_path = s3_experiment_path(
        s3_bucket=orig_params.get('s3_bucket'),
        experiment_name=None if orig_params['no_experiment_subdirectory'] \
            else experiment_name
    )

    if not list_values:
        list_values = [None]
    first_tuner_name, last_tuner_name = None, None
    is_first_iteration = True
    for exp_id, values in enumerate(itertools.product(*list_values)):
        if exp_id < skip_initial_experiments:
            continue
        if num_experiments > 1:
            logger.info(
                "\n---------------------------------\n"
                f"Launching experiment {exp_id} of {num_experiments}\n"
                "---------------------------------")
        if keys:
            extra_dict = dict(zip(keys, values))
            for k in keys:
                if '|' in k:
                    extra_dict.update(zip(k.split('|'), extra_dict[k]))
                    del extra_dict[k]
            params = dict(params_nolistvals, **extra_dict)
        else:
            # Single experiment only
            params = params_nolistvals

        # Select benchmark to run
        benchmark, default_params = benchmark_factory(params)

        # Create scheduler from parameters
        myscheduler, params = scheduler_factory(
            params, benchmark, default_params)

        # Create backend
        backend_name = params['backend']
        if backend_name == 'local':
            logger.info(f"Using 'local' back-end with entry_point = {benchmark['script']}")
            backend = LocalBackend(
                entry_point=benchmark['script'],
                rotate_gpus=params['rotate_gpus'])
        elif backend_name == 'simulated':
            assert benchmark.get('supports_simulated', False), \
                f"Benchmark {params['benchmark_name']} does not support " +\
                "the simulation back-end (has to be tabulated)"
            blackbox_name = benchmark.get('blackbox_name')
            backend_kwargs = dict(
                elapsed_time_attr=benchmark['elapsed_time_attr'],
                tuner_sleep_time=params['tuner_sleep_time'],
                debug_resource_attr=benchmark['resource_attr'])
            if blackbox_name is None:
                logger.info(f"Using 'simulated' back-end with entry_point = {benchmark['script']}")
                # Tabulated benchmark given by a script
                backend_kwargs['entry_point'] = benchmark['script']
                backend = SimulatorBackend(**backend_kwargs)
            else:
                from blackbox_repository.tabulated_benchmark import BlackboxRepositoryBackend

                # Tabulated benchmark from the blackbox repository (simulation
                # runs faster)
                logger.info(f"Using 'simulated' back-end with blackbox_name = {blackbox_name}")
                seed = params.get('blackbox_seed')
                if seed is not None:
                    logger.info(f"Using blackbox with blackbox_seed = {seed}")
                backend_kwargs.update({
                    'blackbox_name': blackbox_name,
                    'dataset': params.get('dataset_name'),
                    'surrogate': benchmark.get('surrogate'),
                    'time_this_resource_attr': benchmark.get(
                        'time_this_resource_attr'),
                    'max_resource_attr': benchmark.get('max_resource_attr'),
                    'seed': seed,
                })
                backend = BlackboxRepositoryBackend(**backend_kwargs)
        else:
            assert backend_name == 'sagemaker'
            for k in ('instance_type',):
                assert params.get(k) is not None, \
                    f"For 'sagemaker' backend, --{k} is needed"
            logger.info(f"Using 'sagemaker' back-end with entry_point = {benchmark['script']}")
            script_path = Path(benchmark['script'])
            sm_estimator = sagemaker_estimator_factory(
                entry_point=script_path.name,
                instance_type=params['instance_type'],
                framework=params.get('framework'),
                role=params.get('sagemaker_execution_role'),
                dependencies=[str(Path(__file__).parent.parent / "benchmarks/")],
                framework_version=params.get('framework_version'),
                pytorch_version=params.get('pytorch_version'),
                source_dir=str(script_path.parent),
                image_uri=params.get('image_uri'),
                disable_profiler=not params['enable_sagemaker_profiler'],
            )
            backend = SagemakerBackend(
                sm_estimator=sm_estimator,
                metrics_names=[benchmark['metric']],
                s3_path=s3_path)

        # Stopping criterion
        num_trials = params.get('num_trials')
        scheduler_timeout = params.get('scheduler_timeout')
        assert not (num_trials is None and scheduler_timeout is None), \
            "One of --num_trials, --scheduler_timeout must be given"
        stop_criterion = StoppingCriterion(
            max_wallclock_time=scheduler_timeout,
            max_num_trials_completed=num_trials)
        if params['no_tuner_logging']:
            # If the tuner does not log anything, we also do not have to
            # compute the status report. This is achieved by setting the
            # update interval large enough
            print_update_interval = 18000 if scheduler_timeout is None \
                else scheduler_timeout + 100
            params['print_update_interval'] = max(
                params['print_update_interval'], print_update_interval)

        # Put together meta-data. Here, we could also do more ...
        metadata = {
            k: v for k, v in params.items() if v is not None}
        for k in ('metric', 'mode', 'resource_attr', 'elapsed_time_attr'):
            if k in benchmark:
                metadata[k] = benchmark[k]
        tuner_name = experiment_name

        try:
            import git

            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            metadata['git_hash'] = sha
            o = repo.remote()
            urls = list(o.urls)
            metadata['git_urls'] = urls
        except Exception:
            pass

        tuner_sleep_time = 0 if backend_name == 'simulated' \
            else params['tuner_sleep_time']

        local_tuner = Tuner(
            backend=backend,
            scheduler=myscheduler,
            stop_criterion=stop_criterion,
            n_workers=params['num_workers'],
            sleep_time=tuner_sleep_time,
            results_update_interval=params['results_update_interval'],
            metadata=metadata,
            tuner_name=tuner_name,
            max_failures=params['max_failures'],
            asynchronous_scheduling=not params['synchronous'],
            print_update_interval=params['print_update_interval'],
            callbacks=[SimulatorCallback()] if backend_name == 'simulated' else None
        )
        last_tuner_name = local_tuner.name
        if is_first_iteration:
            first_tuner_name = copy.copy(last_tuner_name)
            is_first_iteration = False

        if params['local_tuner']:
            # Tuning experiment is run locally
            if params['no_tuner_logging']:
                logging.getLogger('syne_tune.tuner').setLevel(
                    logging.ERROR)
            local_tuner.run()
        else:
            if backend_name != 'sagemaker':
                # Local backend: Configure SageMaker estimator to what the
                # benchmark needs
                instance_type = params['instance_type']
            else:
                # Instance type for tuning, can be different from
                # instance type for workers
                instance_type = params['tuner_instance_type']
            estimator_kwargs = {
                'disable_profiler': not params['enable_sagemaker_profiler']}
            if scheduler_timeout is not None \
                    and scheduler_timeout > 12 * 60 * 60:
                # Make sure that the SageMaker training job running the tuning
                # loop is not stopped before `scheduler_timeout`
                estimator_kwargs['max_run'] = int(1.01 * scheduler_timeout)
            log_level = logging.DEBUG if params['debug_log_level'] \
                else logging.INFO
            root_path = Path(__file__).parent.parent.parent
            dependencies = [
                str(root_path / module)
                for module in ("benchmarking", "blackbox_repository")]
            tuner = RemoteLauncher(
                tuner=local_tuner,
                dependencies=dependencies,
                instance_type=instance_type,
                log_level=log_level,
                s3_path=s3_path,
                no_tuner_logging=params['no_tuner_logging'],
                **estimator_kwargs,
            )
            tuner.run(wait=False)

    logger.info(f"For the record:\n{first_tuner_name} .. {last_tuner_name}")
