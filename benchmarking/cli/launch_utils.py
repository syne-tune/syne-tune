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
import argparse
import logging

from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import \
    SUPPORTED_RESOURCE_FOR_ACQUISITION
from benchmarking.cli.benchmark_factory import supported_benchmarks, benchmark_factory

logger = logging.getLogger(__name__)

__all__ = ['parse_args',
           'make_searcher_and_scheduler',
           ]


def parse_args(allow_lists_as_values=True):
    """
    Argument parser for CLI. Normally, this parameterizes a single experiment.
    But if `allow_lists_as_values == True`, certain arguments admit lists as
    values. In this case, experiments of all combinations of values (Cartesian
    product) are launched.

    :param allow_lists_as_values: See above
    :return: params dict. Note that if an argument added to the parser is not
        provided a value for, it is contained in the dict with value None

    """
    parser = argparse.ArgumentParser(
        description='Asynchronous Hyperparameter Optimization')
    # We parse the CL args twice. The first pass parses all global arguments
    # (not specific to the benchmark). From that pass, we know what the
    # benchmark is. In a second pass, we parse additional benchmark-specific
    # arguments, as defined in the default_params for the benchmark.
    if allow_lists_as_values:
        allow_list = dict(nargs='+')
    else:
        allow_list = dict()

    if allow_lists_as_values:
        parser.add_argument('--argument_groups', type=str,
                            help='Specify groups of list arguments, separated '
                                 'by |. Arguments in a group are iterated '
                                 'over together')
    # Note: The benchmark cannot be a list argument, since it can define its
    # own CL arguments
    parser.add_argument('--benchmark_name', type=str,
                        default='mlp_fashionmnist',
                        choices=supported_benchmarks(),
                        help='Benchmark to run experiment on')
    parser.add_argument('--skip_initial_experiments', type=int, default=0,
                        help='When multiple experiments are launched (due to '
                             'list arguments), this number of initial '
                             'experiments are skipped')
    parser.add_argument('--backend', type=str, default='local',
                        choices=('local', 'sagemaker', 'simulated'),
                        help='Backend for training evaluations')
    parser.add_argument('--local_tuner', action='store_true',
                        help='Run tuning experiment locally? Otherwise, it is '
                             'run remotely (which allows to run multiple '
                             'tuning experiments in parallel)')
    parser.add_argument('--run_id', type=int,
                        help='Identifier to distinguish between runs '
                             '(nonnegative integers)',
                        **allow_list)
    parser.add_argument('--num_runs', type=int,
                        help='Number of repetitions, with run_id 0, 1, ...'
                             'Only if run_id not given (ignored otherwise)')
    parser.add_argument('--random_seed_offset', type=int,
                        help='Master random seed is this plus run_id, modulo '
                             '2 ** 32. Drawn at random if not given')
    parser.add_argument('--instance_type', type=str,
                        help='SageMaker instance type for workers',
                        **allow_list)
    parser.add_argument('--tuner_instance_type', type=str,
                        default='ml.c5.xlarge',
                        help='SageMaker instance type for tuner (only for '
                             'sagemaker backend and remote tuning)',
                        **allow_list)
    parser.add_argument('--num_workers', type=int,
                        help='Number of workers (parallel evaluations)',
                        **allow_list)
    parser.add_argument('--image_uri', type=str,
                        help='URI of Docker image (sagemaker backend)')
    parser.add_argument('--sagemaker_execution_role', type=str,
                        help='SageMaker execution role (sagemaker backend)')
    parser.add_argument('--experiment_name', type=str,
                        help='Experiment name (used as job_name_prefix in '
                             'sagemaker backend)')
    parser.add_argument('--no_debug_log', action='store_true',
                        help='Switch off verbose logging')
    parser.add_argument('--debug_log_level', action='store_true',
                        help='Set logging level to DEBUG (default is INFO)')
    parser.add_argument('--no_tuner_logging', action='store_true',
                        help='By default, the full tuning status is logged '
                             'in the tuning loop every --print_update_interval'
                             ' secs. If this is set, this logging is suppressed')
    parser.add_argument('--enable_sagemaker_profiler', action='store_true',
                        help='Enable SageMaker profiler (this needs one '
                             'processing job for each training job')
    parser.add_argument('--no_experiment_subdirectory', action='store_true',
                        help='When storing results, do not use subdirectory '
                             'experiment_name')
    parser.add_argument('--cost_model_type', type=str,
                        help='Selects cost model of benchmark',
                        **allow_list)
    parser.add_argument('--scheduler', type=str, default='fifo',
                        help='Scheduler name',
                        **allow_list)
    parser.add_argument('--searcher', type=str,
                        help='Searcher name',
                        **allow_list)
    parser.add_argument('--results_update_interval', type=int, default=300,
                        help='Results and tuner state are stored every this '
                             'many seconds')
    parser.add_argument('--print_update_interval', type=int, default=300,
                        help='Tuner status printed every this many seconds')
    parser.add_argument('--tuner_sleep_time', type=float, default=5,
                        help='Tuner tries to fetch new results every this '
                             'many seconds')
    parser.add_argument('--max_resource_level', type=int,
                        help='Largest resource level (e.g., epoch number) '
                             'for training evaluations',
                        **allow_list)
    parser.add_argument('--epochs', type=int,
                        help='Deprecated: Use max_resource_level instead',
                        **allow_list)
    parser.add_argument('--num_trials', type=int,
                        help='Maximum number of trials',
                        **allow_list)
    parser.add_argument('--scheduler_timeout', type=int,
                        help='Trials started until this cutoff time (in secs)',
                        **allow_list)
    parser.add_argument('--max_failures', type=int, default=1,
                        help='The tuning job terminates once this many '
                             'training evaluations failed',
                        **allow_list)
    parser.add_argument('--s3_bucket', type=str,
                        help='S3 bucket to write checkpoints and results to. '
                             'Defaults to default bucket of session')
    parser.add_argument('--no_gpu_rotation', action='store_true',
                        help='For local back-end on a GPU instance: By '
                             'default, trials are launched in parallel '
                             'on different GPU cores (GPU rotation). If '
                             'this is set, all GPU cores are used for a '
                             'single evaluation')
    parser.add_argument('--blackbox_repo_s3_root', type=str,
                        help='S3 root directory for blackbox repository. '
                             'Defaults to default bucket of session')
    parser.add_argument('--blackbox_seed', type=int,
                        help='Fixed seeds of blackbox queries to this value '
                             '(0 is safe), so that they return the same '
                             'metric values for the same config')
    # Arguments for scheduler
    parser.add_argument('--brackets', type=int,
                        help='Number of brackets in HyperbandScheduler',
                        **allow_list)
    parser.add_argument('--reduction_factor', type=float,
                        help='Reduction factor in HyperbandScheduler',
                        **allow_list)
    parser.add_argument('--grace_period', type=int,
                        help='Minimum resource level (e.g., epoch number) '
                             'in HyperbandScheduler',
                        **allow_list)
    parser.add_argument('--rung_levels', type=str,
                        help='List of resource levels to use for the rungs '
                             'in HyperbandScheduler. Entries must be positive '
                             'ints. Overrides --grace_period, '
                             '--reduction_factor if given',
                        **allow_list)
    parser.add_argument('--no_rung_system_per_bracket', action='store_true',
                        help='Parameter of HyperbandScheduler')
    parser.add_argument('--searcher_data', type=str,
                        help='Parameter of HyperbandScheduler',
                        **allow_list)
    parser.add_argument('--register_pending_myopic', action='store_true',
                        help='Parameter of HyperbandScheduler')
    parser.add_argument('--not_normalize_targets', action='store_true',
                        help='Do not normalize targets to mean 0, variance 1'
                             ' before fitting surrogate model')
    parser.add_argument('--pasha_ranking_criterion', type=str,
                        help='Parameter of PASHA scheduler',
                        **allow_list)
    parser.add_argument('--pasha_epsilon', type=float,
                        help='Parameter of PASHA scheduler',
                        **allow_list)
    parser.add_argument('--pasha_epsilon_scaling', type=str,
                        help='Parameter of PASHA scheduler',
                        **allow_list)
    # Arguments for bayesopt searcher
    parser.add_argument('--searcher_model', type=str,
                        help='Surrogate model for bayesopt searcher with '
                             'HyperbandScheduler',
                        **allow_list)
    parser.add_argument('--searcher_num_init_random', type=int,
                        help='Number of initial trials not chosen by searcher',
                        **allow_list)
    parser.add_argument('--searcher_num_init_candidates', type=int,
                        help='Number of random candidates scored to seed search',
                        **allow_list)
    parser.add_argument('--searcher_num_fantasy_samples', type=int,
                        help='Number of fantasy samples',
                        **allow_list)
    help_str = "Rule for resource level at which acquisition function is used " +\
        f"[{SUPPORTED_RESOURCE_FOR_ACQUISITION}]"
    parser.add_argument('--searcher_resource_acq', type=str,
                        help=help_str,
                        **allow_list)
    parser.add_argument('--searcher_resource_acq_bohb_threshold', type=int,
                        help='Parameter for resource_acq == bohb',
                        **allow_list)
    parser.add_argument('--searcher_gp_resource_kernel', type=str,
                        help='Multi-task kernel for HyperbandScheduler',
                        **allow_list)
    parser.add_argument('--searcher_opt_skip_period', type=int,
                        help='Update GP hyperparameters only every (...) times',
                        **allow_list)
    parser.add_argument('--searcher_opt_skip_init_length', type=int,
                        help='Update GP hyperparameters every time until '
                             '(...) observations are done',
                        **allow_list)
    parser.add_argument('--searcher_opt_skip_num_max_resource',
                        action='store_true',
                        help='Update GP hyperparameters only when training '
                             'runs reach max_t')
    parser.add_argument('--searcher_opt_nstarts', type=int,
                        help='GP hyperparameter optimization restarted (...) '
                             'times',
                        **allow_list)
    parser.add_argument('--searcher_opt_maxiter', type=int,
                        help='Maximum number of iterations of GP '
                             'hyperparameter optimization',
                        **allow_list)
    parser.add_argument('--searcher_initial_scoring', type=str,
                        help='Scoring function to rank initial candidates '
                             'for seeding search [thompson_indep, acq_func]',
                        **allow_list)
    parser.add_argument('--searcher_issm_gamma_one', action='store_true',
                        help='Fix gamma parameter of ISSM to one?')
    parser.add_argument('--searcher_exponent_cost', type=float,
                        help='Exponent of cost term in cost-aware expected '
                             'improvement acquisition function',
                        **allow_list)
    parser.add_argument('--searcher_expdecay_normalize_inputs', action='store_true',
                        help='Normalize resource values to [0, 1] in '
                             'GP-expdecay surrogate model (only if '
                             'searcher_model = gp_expdecay)')
    parser.add_argument('--searcher_num_init_candidates_for_batch', type=int,
                        help='Relevant for synchronous Hyperband with bayesopt '
                             'searcher. If batch of size B is suggested, the '
                             'first suggest uses searcher_num_init_candidates, '
                             'the B-1 subsequent suggests use this value',
                        **allow_list)
    parser.add_argument('--searcher_use_old_code',
                        action='store_true',
                        help='DEBUG: Use old code for gp_issm, gp_expdecay')
    parser.add_argument('--searcher_no_fantasizing', action='store_true',
                        help='Ignore pending evaluations, do not use fantasizing')
    # Arguments for kde searcher
    parser.add_argument('--searcher_num_min_data_points', type=int,
                        help='KDE: Minimum number of datapoints needed to fit models',
                        **allow_list)
    parser.add_argument('--searcher_top_n_percent', type=int,
                        help='KDE: Top (bottom) model fit on this top (bottom) fraction of data',
                        **allow_list)
    parser.add_argument('--searcher_min_bandwidth', type=float,
                        help='KDE: Minimum bandwidth',
                        **allow_list)
    parser.add_argument('--searcher_num_candidates', type=int,
                        help='KDE: Number of candidates that are sampled to optimize the acquisition function',
                        **allow_list)
    parser.add_argument('--searcher_bandwidth_factor', type=int,
                        help='KDE: Parameter to scale bandwidth',
                        **allow_list)
    parser.add_argument('--searcher_random_fraction', type=float,
                        help='KDE: Fraction of configs suggested at random',
                        **allow_list)

    # First pass: All global arguments
    # Why do we parse all global args here, and not just benchmark_name?
    # This is to make sure that the help option of the parser lists all
    # global arguments and their help strings.
    _params = parser.parse_known_args()[0]
    benchmark_name = _params.benchmark_name

    # Add benchmark-specific CL args (if any)
    # These are the ones listed in benchmark['default_params'], minus args which
    # are already global (i.e., added above)
    _, default_params = benchmark_factory({'benchmark_name': benchmark_name})
    help_str = f"Additional parameter for {benchmark_name} benchmark"
    have_extra_args = False
    for name, value in default_params.items():
        try:
            # We don't need to set defaults here
            if value is None:
                _type = str
            else:
                _type = type(value)
            parser.add_argument('--' + name, type=_type, help=help_str)
            have_extra_args = True
        except argparse.ArgumentError:
            pass

    # Second pass: All args (global and benchmark-specific)
    if have_extra_args:
        params = vars(parser.parse_args())
    else:
        params = _params
    # Post-processing
    params['debug_log'] = not params['no_debug_log']
    del params['no_debug_log']
    params['rotate_gpus'] = not params['no_gpu_rotation']
    del params['no_gpu_rotation']
    epochs = params.get('epochs')
    if params.get('max_resource_level') is None:
        if epochs is not None:
            logger.info("--epochs is deprecated, please use "
                        "--max_resource_level in the future")
            params['max_resource_level'] = epochs
    elif epochs is not None:
        logger.info("Both --max_resource_level and the deprecated "
                    "--epochs are set. The latter is ignored")
    if 'epochs' in params:
        del params['epochs']
    params['rung_system_per_bracket'] = not params['no_rung_system_per_bracket']
    del params['no_rung_system_per_bracket']
    params['normalize_targets'] = not params['not_normalize_targets']
    del params['not_normalize_targets']
    params['searcher_use_new_code'] = not params['searcher_use_old_code']
    del params['searcher_use_old_code']
    return params


def _enter_not_none(dct, key, val, tp=None):
    if tp is None:
        tp = str
    if val is not None:
        dct[key] = tp(val)


def make_searcher_and_scheduler(params) -> (dict, dict):
    scheduler = params['scheduler']
    searcher = params['searcher']
    # Options for searcher
    search_options = dict()
    _enter_not_none(
        search_options, 'debug_log', params.get('debug_log'), tp=bool)
    _enter_not_none(
        search_options, 'normalize_targets', params.get('normalize_targets'),
        tp=bool)
    model = params.get('searcher_model')
    _enter_not_none(search_options, 'model', model)

    if searcher.startswith('bayesopt'):
        # Options for bayesopt searcher
        searcher_args = (
            ('num_init_random', int, False),
            ('num_init_candidates', int, False),
            ('num_fantasy_samples', int, False),
            ('resource_acq', str, True),
            ('resource_acq_bohb_threshold', int, True),
            ('gp_resource_kernel', str, True),
            ('opt_skip_period', int, False),
            ('opt_skip_init_length', int, False),
            ('opt_skip_num_max_resource', bool, False),
            ('opt_nstarts', int, False),
            ('opt_maxiter', int, False),
            ('initial_scoring', str, False),
            ('issm_gamma_one', bool, False),
            ('exponent_cost', float, False),
            ('expdecay_normalize_inputs', bool, False),
            ('use_new_code', bool, False),
            ('num_init_candidates_for_batch', int, False),
            ('no_fantasizing', bool, False),
        )
        gp_add_models = {'gp_issm', 'gp_expdecay'}
        for name, tp, warn in searcher_args:
            _enter_not_none(
                search_options, name, params.get('searcher_' + name), tp=tp)
            if warn and name in search_options and model in gp_add_models:
                logger.warning(f"{name} not used with searcher_model = {model}")
        if 'issm_gamma_one' in search_options and model != 'gp_issm':
            logger.warning(
                f"searcher_issm_gamma_one not used with searcher_model = {model}")
        if 'expdecay_normalize_inputs' in search_options and model != 'gp_expdecay':
            logger.warning(
                "searcher_expdecay_normalize_inputs not used with searcher_model "
                f"= {model}")
    elif searcher == 'kde':
        # Options for kde searcher
        searcher_args = (
            ('num_min_data_points', int),
            ('top_n_percent', int),
            ('min_bandwidth', float),
            ('num_candidates', int),
            ('bandwidth_factor', int),
            ('random_fraction', float),
        )
        for name, tp in searcher_args:
            _enter_not_none(
                search_options, name, params.get('searcher_' + name), tp=tp)

    # Options for scheduler
    random_seed_offset = params.get('random_seed_offset')
    if random_seed_offset is None:
        random_seed_offset = 0
    random_seed = (random_seed_offset + params['run_id']) % (2 ** 32)
    scheduler_options = {'random_seed': random_seed}
    name = 'max_resource_level' if scheduler == 'hyperband_synchronous' \
        else 'max_t'
    _enter_not_none(
        scheduler_options, name, params.get('max_resource_level'), tp=int)
    scheduler_args = ()
    if scheduler != 'fifo':
        # Only process these arguments for HyperbandScheduler
        prefix = 'hyperband_'
        assert scheduler.startswith(prefix)
        scheduler_args = scheduler_args + (
            ('reduction_factor', int),
            ('grace_period', int),
            ('brackets', int))
        if scheduler != 'hyperband_synchronous':
            sch_type = scheduler[len(prefix):]
            _enter_not_none(scheduler_options, 'type', sch_type)
            rung_levels = params.get('rung_levels')
            if rung_levels is not None:
                scheduler_options['rung_levels'] = sorted(
                    [int(x) for x in rung_levels.split()])
            scheduler_args = scheduler_args + (
                ('searcher_data', str),
                ('register_pending_myopic', bool),
                ('rung_system_per_bracket', bool))
    for name, tp in scheduler_args:
        _enter_not_none(
            scheduler_options, name, params.get(name), tp=tp)

    # Special constraints
    if scheduler != 'fifo' and searcher.startswith('bayesopt') \
            and model in gp_add_models:
        searcher_data = scheduler_options.get('searcher_data')
        if searcher_data is not None and searcher_data != 'all':
            logger.warning(
                f"searcher_model = '{model}' requires "
                f"searcher_data = 'all' (and not '{searcher_data}')")
        scheduler_options['searcher_data'] = 'all'

    return search_options, scheduler_options
