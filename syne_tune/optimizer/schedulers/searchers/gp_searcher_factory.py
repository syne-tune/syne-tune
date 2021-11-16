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
from typing import Set, Dict
import logging

from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils \
    import map_reward_const_minus_x, MapReward, \
    DEFAULT_INITIAL_SCORING, SUPPORTED_INITIAL_SCORING, \
    ResourceForAcquisitionBOHB, ResourceForAcquisitionFirstMilestone, \
    ResourceForAcquisitionMap
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.kernel_factory \
    import resource_kernel_factory
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants \
    import OptimizationConfig, DEFAULT_OPTIMIZATION_CONFIG
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_regression \
    import GaussianProcessRegression
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import Matern52
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean \
    import ScalarMeanFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.model_params \
    import IndependentISSModelParameters
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.gpiss_model \
    import GaussianProcessISSModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_skipopt \
    import SkipNoMaxResourcePredicate, SkipPeriodicallyPredicate
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model \
    import GaussProcEmpiricalBayesModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gpiss_model \
    import GaussProcISSModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost_fifo_model \
    import CostSurrogateModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl \
    import EIAcquisitionFunction, CEIAcquisitionFunction, EIpuAcquisitionFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults \
    import DEFAULT_NUM_INITIAL_CANDIDATES, DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import INTERNAL_METRIC_NAME, \
    INTERNAL_CONSTRAINT_NAME, INTERNAL_COST_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log \
    import DebugLogPrinter
from syne_tune.optimizer.schedulers.utils.simple_profiler \
    import SimpleProfiler
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import Integer, Categorical, Boolean, Float

__all__ = ['gp_fifo_searcher_factory',
           'gp_multifidelity_searcher_factory',
           'constrained_gp_fifo_searcher_factory',
           'cost_aware_coarse_gp_fifo_searcher_factory',
           'cost_aware_fine_gp_fifo_searcher_factory',
           'cost_aware_gp_multifidelity_searcher_factory',
           'gp_fifo_searcher_defaults',
           'gp_multifidelity_searcher_defaults',
           'constrained_gp_fifo_searcher_defaults',
           'cost_aware_gp_fifo_searcher_defaults',
           'cost_aware_gp_multifidelity_searcher_defaults',
           ]

logger = logging.getLogger(__name__)


def _create_gp_common(hp_ranges_cs, **kwargs):
    opt_warmstart = kwargs.get('opt_warmstart', False)
    kernel = Matern52(dimension=hp_ranges_cs.ndarray_size(), ARD=True)
    mean = ScalarMeanFunction()
    optimization_config = OptimizationConfig(
        lbfgs_tol=DEFAULT_OPTIMIZATION_CONFIG.lbfgs_tol,
        lbfgs_maxiter=kwargs['opt_maxiter'],
        verbose=kwargs['opt_verbose'],
        n_starts=kwargs['opt_nstarts'])
    if kwargs.get('profiler', False):
        profiler = SimpleProfiler()
    else:
        profiler = None
    if kwargs.get('debug_log', False):
        debug_log = DebugLogPrinter()
    else:
        debug_log = None
    return opt_warmstart, kernel, mean, optimization_config, profiler, \
           debug_log


def _create_gp_standard_model(
        hp_ranges_cs, active_metric, random_seed, is_hyperband=False,
        **kwargs):
    opt_warmstart, kernel, mean, optimization_config, profiler, debug_log = \
        _create_gp_common(hp_ranges_cs, **kwargs)
    if is_hyperband:
        kernel, mean = resource_kernel_factory(
            kwargs['gp_resource_kernel'],
            kernel_x=kernel, mean_x=mean)
    gpmodel = GaussianProcessRegression(
        kernel=kernel, mean=mean, optimization_config=optimization_config,
        random_seed=random_seed, fit_reset_params=not opt_warmstart)
    model_factory = GaussProcEmpiricalBayesModelFactory(
        active_metric=active_metric,
        gpmodel=gpmodel,
        num_fantasy_samples=kwargs['num_fantasy_samples'],
        normalize_targets=True,
        profiler=profiler,
        debug_log=debug_log)
    return model_factory


def _create_gpiss_model(hp_ranges_cs, active_metric, random_seed, **kwargs):
    opt_warmstart, kernel, mean, optimization_config, profiler, debug_log = \
        _create_gp_common(hp_ranges_cs, **kwargs)
    iss_model = IndependentISSModelParameters(
        gamma_is_one=kwargs.get('issm_gamma_one', False))
    gpmodel = GaussianProcessISSModel(
        kernel=kernel, iss_model=iss_model, mean=mean,
        optimization_config=optimization_config, random_seed=random_seed,
        fit_reset_params=not opt_warmstart)
    model_factory = GaussProcISSModelFactory(
        active_metric=active_metric, gpmodel=gpmodel, profiler=profiler,
        debug_log=debug_log)
    return model_factory


def _create_common_objects(is_issm=False, is_multi_output=False, **kwargs):
    scheduler = kwargs['scheduler']
    config_space = kwargs['configspace']
    is_hyperband = scheduler.startswith('hyperband')
    assert (not is_issm) or is_hyperband, \
        "GP-ISS model only together with hyperband_* scheduler"
    assert (not is_multi_output) or (not is_hyperband), \
        "MultiOutput BO only together with FIFO scheduler"
    hp_ranges = make_hyperparameter_ranges(config_space)
    key = 'random_seed_generator'
    if key in kwargs:
        rs_generator = kwargs[key]
        random_seed1 = rs_generator()
        random_seed2 = rs_generator()
        _kwargs = {k: v for k, v in kwargs.items() if k != key}
    else:
        key = 'random_seed'
        if key in kwargs:
            random_seed1 = kwargs[key]
            _kwargs = {k: v for k, v in kwargs.items() if k != key}
        else:
            random_seed1 = 31415927
            _kwargs = kwargs
        random_seed2 = random_seed1
    # Skip optimization predicate for GP surrogate model
    if kwargs.get('opt_skip_num_max_resource', False) and is_hyperband:
        skip_optimization = SkipNoMaxResourcePredicate(
            init_length=kwargs['opt_skip_init_length'],
            max_resource=kwargs['max_epochs'])
    elif kwargs.get('opt_skip_period', 1) > 1:
        skip_optimization = SkipPeriodicallyPredicate(
            init_length=kwargs['opt_skip_init_length'],
            period=kwargs['opt_skip_period'])
    else:
        skip_optimization = None
    # Conversion from reward to metric (strictly decreasing) and back.
    # This is done only if the scheduler mode is 'max'.
    scheduler_mode = kwargs.get('scheduler_mode', 'min')
    if scheduler_mode == 'max':
        _map_reward = kwargs.get('map_reward', '1_minus_x')
        if isinstance(_map_reward, str):
            _map_reward_name = _map_reward
            assert _map_reward_name.endswith('minus_x'), \
                f"map_reward = {_map_reward_name} is not supported (use " + \
                "'minus_x' or '*_minus_x')"
            if _map_reward_name == 'minus_x':
                const = 0.0
            else:
                # Allow strings '*_minus_x', parse const for *
                # Example: '1_minus_x' => const = 1
                offset = len(_map_reward_name) - len('_minus_x')
                const = float(_map_reward_name[:offset])
            _map_reward: MapReward = map_reward_const_minus_x(const=const)
        else:
            assert isinstance(_map_reward, MapReward), \
                "map_reward must either be string or of MapReward type"
    else:
        assert scheduler_mode == 'min', \
            f"scheduler_mode = {scheduler_mode}, must be in ('max', 'min')"
        _map_reward = kwargs.get('map_reward')
        if _map_reward is not None:
            logger.warning(
                f"Since scheduler_mode == 'min', map_reward = {_map_reward} is ignored")
            _map_reward = None

    # Create model factory
    if not is_issm:
        model_factory = _create_gp_standard_model(
            hp_ranges_cs=hp_ranges,
            active_metric=INTERNAL_METRIC_NAME,
            random_seed=random_seed1,
            is_hyperband=is_hyperband,
            **_kwargs)
    else:
        model_factory = _create_gpiss_model(
            hp_ranges_cs=hp_ranges,
            active_metric=INTERNAL_METRIC_NAME,
            random_seed=random_seed1,
            **_kwargs)

    return hp_ranges, random_seed2, model_factory, _map_reward, \
           skip_optimization


def gp_fifo_searcher_factory(**kwargs) -> Dict:
    """
    Returns kwargs for `GPFIFOSearcher._create_internal`, based on kwargs
    equal to search_options passed to and extended by scheduler (see
    :class:`FIFOScheduler`).

    Extensions of kwargs by the scheduler:
    - scheduler: Name of scheduler ('fifo', 'hyperband_*')
    - configspace: Configuration space
    Only Hyperband schedulers:
    - resource_attr: Name of resource (or time) attribute
    - max_epochs: Maximum resource value

    :param kwargs: search_options coming from scheduler
    :return: kwargs for GPFIFOSearcher._create_internal

    """
    assert kwargs['scheduler'] in ['fifo'], \
        "This factory needs scheduler = 'fifo' (instead of '{}')".format(
            kwargs['scheduler'])
    # Common objects
    hp_ranges, random_seed, model_factory, _map_reward, skip_optimization \
        = _create_common_objects(**kwargs)

    return dict(
        hp_ranges=hp_ranges,
        random_seed=random_seed,
        model_factory=model_factory,
        map_reward=_map_reward,
        acquisition_class=EIAcquisitionFunction,
        skip_optimization=skip_optimization,
        num_initial_candidates=kwargs['num_init_candidates'],
        num_initial_random_choices=kwargs['num_init_random'],
        initial_scoring=kwargs['initial_scoring'],
        cost_attr=kwargs['cost_attr'])


def _resource_for_acquisition(
        kwargs: Dict,
        hp_ranges: HyperparameterRanges) -> ResourceForAcquisitionMap:
    _resource_acq = kwargs.get('resource_acq', 'bohb')
    if _resource_acq == 'bohb':
        threshold = kwargs.get(
            'resource_acq_bohb_threshold', hp_ranges.ndarray_size())
        resource_for_acquisition = ResourceForAcquisitionBOHB(
            threshold=threshold)
    else:
        assert _resource_acq == 'first', \
            "resource_acq must be 'bohb' or 'first'"
        resource_for_acquisition = ResourceForAcquisitionFirstMilestone()
    return resource_for_acquisition


def gp_multifidelity_searcher_factory(**kwargs) -> Dict:
    """
    Returns kwargs for `GPMultiFidelitySearcher._create_internal`, based on
    kwargs equal to search_options passed to and extended by scheduler (see
    :class:`HyperbandScheduler`).

    :param kwargs: search_options coming from scheduler
    :return: kwargs for GPMultiFidelitySearcher._create_internal

    """
    supp_schedulers = {'hyperband_stopping', 'hyperband_promotion'}
    assert kwargs['scheduler'] in supp_schedulers, \
        "This factory needs scheduler in {} (instead of '{}')".format(
            supp_schedulers, kwargs['scheduler'])
    # Common objects
    hp_ranges, random_seed, model_factory, _map_reward, skip_optimization \
        = _create_common_objects(**kwargs)

    resource_for_acquisition = _resource_for_acquisition(kwargs, hp_ranges)
    epoch_range = (1, kwargs['max_epochs'])
    return dict(
        hp_ranges=hp_ranges,
        resource_attr=kwargs['resource_attr'],
        resource_attr_range=epoch_range,
        random_seed=random_seed,
        model_factory=model_factory,
        map_reward=_map_reward,
        acquisition_class=EIAcquisitionFunction,
        resource_for_acquisition=resource_for_acquisition,
        skip_optimization=skip_optimization,
        num_initial_candidates=kwargs['num_init_candidates'],
        num_initial_random_choices=kwargs['num_init_random'],
        initial_scoring=kwargs['initial_scoring'],
        cost_attr=kwargs['cost_attr'])


def gpiss_multifidelity_searcher_factory(**kwargs) -> Dict:
    """
    Returns kwargs for `GPISSMultiFidelitySearcher._create_internal`, based on
    kwargs equal to search_options passed to and extended by scheduler (see
    :class:`HyperbandScheduler`).

    Note: The model factory in `kwargs['model_factory']` still lacks its
    `ExtendedConfiguration` space, which must be set with `set_configspace_ext`
    before the factory can be used.

    :param kwargs: search_options coming from scheduler
    :return: kwargs for GPISSMultiFidelitySearcher._create_internal

    """
    supp_schedulers = {'hyperband_stopping', 'hyperband_promotion'}
    assert kwargs['scheduler'] in supp_schedulers, \
        "This factory needs scheduler in {} (instead of '{}')".format(
            supp_schedulers, kwargs['scheduler'])
    # Common objects
    hp_ranges, random_seed, model_factory, _map_reward, skip_optimization \
        = _create_common_objects(is_issm=True, **kwargs)

    epoch_range = (1, kwargs['max_epochs'])
    return dict(
        hp_ranges=hp_ranges,
        resource_attr=kwargs['resource_attr'],
        resource_attr_range=epoch_range,
        random_seed=random_seed,
        model_factory=model_factory,
        map_reward=_map_reward,
        acquisition_class=EIAcquisitionFunction,
        skip_optimization=skip_optimization,
        num_initial_candidates=kwargs['num_init_candidates'],
        num_initial_random_choices=kwargs['num_init_random'],
        initial_scoring=kwargs['initial_scoring'],
        cost_attr=kwargs['cost_attr'])


def constrained_gp_fifo_searcher_factory(**kwargs) -> Dict:
    """
    Returns kwargs for `ConstrainedGPFIFOSearcher._create_internal`, based on kwargs
    equal to search_options passed to and extended by scheduler (see
    :class:`FIFOScheduler`).

    :param kwargs: search_options coming from scheduler
    :return: kwargs for ConstrainedGPFIFOSearcher._create_internal

    """
    assert kwargs['scheduler'] in ['fifo'], \
        "This factory needs scheduler = 'fifo' (instead of '{}')".format(
            kwargs['scheduler'])
    # Common objects
    hp_ranges, random_seed, model_factory, _map_reward, skip_optimization \
        = _create_common_objects(**kwargs)
    # We need two model factories: one for active metric (model_factory),
    # the other for constraint metric (model_factory_constraint)
    key = 'random_seed'
    if key in kwargs:
        _kwargs = {k: v for k, v in kwargs.items() if k != key}
    else:
        _kwargs = kwargs
    model_factory_constraint = _create_gp_standard_model(
        hp_ranges_cs=hp_ranges,
        active_metric=INTERNAL_CONSTRAINT_NAME,
        random_seed=random_seed,
        **_kwargs)
    # Sharing debug_log attribute across models
    model_factory_constraint._debug_log = model_factory._debug_log
    # The same skip_optimization strategy applies to both models
    skip_optimization_constraint = skip_optimization

    output_model_factory = {INTERNAL_METRIC_NAME: model_factory,
                            INTERNAL_CONSTRAINT_NAME: model_factory_constraint}
    output_skip_optimization = {INTERNAL_METRIC_NAME: skip_optimization,
                                INTERNAL_CONSTRAINT_NAME: skip_optimization_constraint}

    return dict(
        hp_ranges=hp_ranges,
        random_seed=random_seed,
        output_model_factory=output_model_factory,
        map_reward=_map_reward,
        acquisition_class=CEIAcquisitionFunction,
        output_skip_optimization=output_skip_optimization,
        num_initial_candidates=kwargs['num_init_candidates'],
        num_initial_random_choices=kwargs['num_init_random'],
        initial_scoring=kwargs['initial_scoring'],
        cost_attr=kwargs['cost_attr'])


def cost_aware_coarse_gp_fifo_searcher_factory(**kwargs) -> Dict:
    """
    Returns kwargs for `CostAwareGPFIFOSearcher._create_internal`, based on
    kwargs equal to search_options passed to and extended by scheduler (see
    :class:`FIFOScheduler`).
    This is for the coarse-grained variant, where costs c(x) are obtained
    together with metric values and are given a GP surrogate model.

    :param kwargs: search_options coming from scheduler
    :return: kwargs for CostAwareGPFIFOSearcher._create_internal

    """
    assert kwargs['scheduler'] in ['fifo'], \
        "This factory needs scheduler = 'fifo' (instead of '{}')".format(
            kwargs['scheduler'])
    # Common objects
    hp_ranges, random_seed, model_factory, _map_reward, skip_optimization \
        = _create_common_objects(**kwargs)
    # We need two model factories: one for active metric (model_factory),
    # the other for cost metric (model_factory_cost)
    key = 'random_seed'
    if key in kwargs:
        _kwargs = {k: v for k, v in kwargs.items() if k != key}
    else:
        _kwargs = kwargs
    model_factory_cost = _create_gp_standard_model(
        hp_ranges_cs=hp_ranges,
        active_metric=INTERNAL_COST_NAME,
        random_seed=random_seed,
        **_kwargs)
    # Sharing debug_log attribute across models
    model_factory_cost._debug_log = model_factory._debug_log
    exponent_cost = kwargs.get('exponent_cost', 1.0)
    acquisition_class = (
        EIpuAcquisitionFunction, dict(exponent_cost=exponent_cost))
    # The same skip_optimization strategy applies to both models
    skip_optimization_cost = skip_optimization

    output_model_factory = {INTERNAL_METRIC_NAME: model_factory,
                            INTERNAL_COST_NAME: model_factory_cost}
    output_skip_optimization = {INTERNAL_METRIC_NAME: skip_optimization,
                                INTERNAL_COST_NAME: skip_optimization_cost}

    return dict(
        hp_ranges=hp_ranges,
        random_seed=random_seed,
        output_model_factory=output_model_factory,
        map_reward=_map_reward,
        acquisition_class=acquisition_class,
        output_skip_optimization=output_skip_optimization,
        num_initial_candidates=kwargs['num_init_candidates'],
        num_initial_random_choices=kwargs['num_init_random'],
        initial_scoring=kwargs['initial_scoring'],
        cost_attr=kwargs['cost_attr'])


def cost_aware_fine_gp_fifo_searcher_factory(**kwargs) -> Dict:
    """
    Returns kwargs for `CostAwareGPFIFOSearcher._create_internal`, based on
    kwargs equal to search_options passed to and extended by scheduler (see
    :class:`FIFOScheduler`).
    This is for the fine-grained variant, where costs c(x, r) are obtained
    with each report and are represented by a :class:`CostModel`
    surrogate model.

    :param kwargs: search_options coming from scheduler
    :return: kwargs for CostAwareGPFIFOSearcher._create_internal

    """
    assert kwargs['scheduler'] in ['fifo'], \
        "This factory needs scheduler = 'fifo' (instead of '{}')".format(
            kwargs['scheduler'])
    cost_model = kwargs.get('cost_model')
    assert cost_model is not None, \
        "If search_options['resource_attr'] is given, a CostModel has " +\
        "to be specified in search_options['cost_model']"
    fixed_resource = kwargs.get('max_epochs')
    assert fixed_resource is not None, \
        "If search_options['resource_attr'] is given, the maximum " +\
        "resource level has to be specified in " +\
        "search_options['max_epochs'], or (simpler) as max_t when " +\
        "creating FIFOScheduler"
    # Common objects
    hp_ranges, random_seed, model_factory, _map_reward, skip_optimization \
        = _create_common_objects(**kwargs)
    # We need two model factories: one for active metric (model_factory),
    # the other for cost metric (model_factory_cost)
    model_factory_cost = CostSurrogateModelFactory(
        model=kwargs['cost_model'],
        fixed_resource=fixed_resource,
        num_samples=1)
    exponent_cost = kwargs.get('exponent_cost', 1.0)
    acquisition_class = (
        EIpuAcquisitionFunction, dict(exponent_cost=exponent_cost))
    # The same skip_optimization strategy applies to both models
    skip_optimization_cost = skip_optimization

    output_model_factory = {INTERNAL_METRIC_NAME: model_factory,
                            INTERNAL_COST_NAME: model_factory_cost}
    output_skip_optimization = {INTERNAL_METRIC_NAME: skip_optimization,
                                INTERNAL_COST_NAME: skip_optimization_cost}

    return dict(
        hp_ranges=hp_ranges,
        random_seed=random_seed,
        output_model_factory=output_model_factory,
        map_reward=_map_reward,
        acquisition_class=acquisition_class,
        output_skip_optimization=output_skip_optimization,
        num_initial_candidates=kwargs['num_init_candidates'],
        num_initial_random_choices=kwargs['num_init_random'],
        initial_scoring=kwargs['initial_scoring'],
        cost_attr=kwargs['cost_attr'],
        resource_attr=kwargs['resource_attr'])


def cost_aware_gp_multifidelity_searcher_factory(**kwargs) -> Dict:
    """
    Returns kwargs for `CostAwareGPMultiFidelitySearcher._create_internal`,
    based on kwargs equal to search_options passed to and extended by
    scheduler (see :class:`HyperbandScheduler`).

    :param kwargs: search_options coming from scheduler
    :return: kwargs for CostAwareGPMultiFidelitySearcher._create_internal

    """
    supp_schedulers = {'hyperband_stopping', 'hyperband_promotion'}
    assert kwargs['scheduler'] in supp_schedulers, \
        "This factory needs scheduler in {} (instead of '{}')".format(
            supp_schedulers, kwargs['scheduler'])
    cost_model = kwargs.get('cost_model')
    assert cost_model is not None, \
        "If search_options['resource_attr'] is given, a CostModel has " +\
        "to be specified in search_options['cost_model']"
    # Common objects
    hp_ranges, random_seed, model_factory, _map_reward, skip_optimization \
        = _create_common_objects(**kwargs)
    # We need two model factories: one for active metric (model_factory),
    # the other for cost metric (model_factory_cost)
    model_factory_cost = CostSurrogateModelFactory(
        model=kwargs['cost_model'],
        fixed_resource=kwargs['max_epochs'],
        num_samples=1)
    exponent_cost = kwargs.get('exponent_cost', 1.0)
    acquisition_class = (
        EIpuAcquisitionFunction, dict(exponent_cost=exponent_cost))
    # The same skip_optimization strategy applies to both models
    skip_optimization_cost = skip_optimization

    output_model_factory = {INTERNAL_METRIC_NAME: model_factory,
                            INTERNAL_COST_NAME: model_factory_cost}
    output_skip_optimization = {INTERNAL_METRIC_NAME: skip_optimization,
                                INTERNAL_COST_NAME: skip_optimization_cost}

    resource_for_acquisition = _resource_for_acquisition(kwargs, hp_ranges)
    epoch_range = (1, kwargs['max_epochs'])
    return dict(
        hp_ranges=hp_ranges,
        resource_attr=kwargs['resource_attr'],
        resource_attr_range=epoch_range,
        random_seed=random_seed,
        output_model_factory=output_model_factory,
        map_reward=_map_reward,
        acquisition_class=acquisition_class,
        resource_for_acquisition=resource_for_acquisition,
        output_skip_optimization=output_skip_optimization,
        num_initial_candidates=kwargs['num_init_candidates'],
        num_initial_random_choices=kwargs['num_init_random'],
        initial_scoring=kwargs['initial_scoring'],
        cost_attr=kwargs['cost_attr'])


def _common_defaults(is_hyperband: bool, is_multi_output: bool) -> (Set[str], dict, dict):
    mandatory = set()

    default_options = {
        'opt_skip_init_length': 150,
        'opt_skip_period': 1,
        'profiler': False,
        'opt_maxiter': 50,
        'opt_nstarts': 2,
        'opt_warmstart': False,
        'opt_verbose': False,
        'opt_debug_writer': False,
        'num_fantasy_samples': 20,
        'scheduler': 'fifo',
        'num_init_random': DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
        'num_init_candidates': DEFAULT_NUM_INITIAL_CANDIDATES,
        'initial_scoring': DEFAULT_INITIAL_SCORING,
        'debug_log': True,
        'cost_attr': 'elapsed_time'}
    if is_hyperband:
        default_options['opt_skip_num_max_resource'] = False
        default_options['gp_resource_kernel'] = 'exp-decay-sum'
        default_options['resource_acq'] = 'bohb'
        default_options['resource_acq_bohb_threshold'] = 3
        default_options['num_init_random'] = 6
        default_options['issm_gamma_one'] = False
    if is_multi_output:
        default_options['initial_scoring'] = 'acq_func'
        default_options['exponent_cost'] = 1.0

    constraints = {
        'random_seed': Integer(0, 2 ** 32 - 1),
        'opt_skip_init_length': Integer(0, None),
        'opt_skip_period': Integer(1, None),
        'profiler': Boolean(),
        'opt_maxiter': Integer(1, None),
        'opt_nstarts': Integer(1, None),
        'opt_warmstart': Boolean(),
        'opt_verbose': Boolean(),
        'opt_debug_writer': Boolean(),
        'num_fantasy_samples': Integer(1, None),
        'num_init_random': Integer(0, None),
        'num_init_candidates': Integer(5, None),
        'initial_scoring': Categorical(
            choices=tuple(SUPPORTED_INITIAL_SCORING)),
        'debug_log': Boolean()}
    if is_hyperband:
        constraints['opt_skip_num_max_resource'] = Boolean()
        constraints['gp_resource_kernel'] = Categorical(choices=(
            'exp-decay-sum', 'exp-decay-combined', 'exp-decay-delta1',
            'matern52', 'matern52-res-warp'))
        constraints['resource_acq'] = Categorical(
            choices=('bohb', 'first'))
        constraints['issm_gamma_one'] = Boolean()
    if is_multi_output:
        constraints['initial_scoring'] = Categorical(
            choices=tuple({'acq_func'}))
        constraints['exponent_cost'] = Float(0.0, 1.0)

    return mandatory, default_options, constraints


def gp_fifo_searcher_defaults() -> (Set[str], dict, dict):
    """
    Returns mandatory, default_options, config_space for
    check_and_merge_defaults to be applied to search_options for
    :class:`GPFIFOSearcher`.

    :return: (mandatory, default_options, config_space)

    """
    return _common_defaults(is_hyperband=False, is_multi_output=False)


def constrained_gp_fifo_searcher_defaults() -> (Set[str], dict, dict):
    """
    Returns mandatory, default_options, config_space for
    check_and_merge_defaults to be applied to search_options for
    :class:`ConstrainedGPFIFOSearcher`.

    :return: (mandatory, default_options, config_space)

    """
    return _common_defaults(is_hyperband=False, is_multi_output=True)


def cost_aware_gp_fifo_searcher_defaults() -> (Set[str], dict, dict):
    """
    Returns mandatory, default_options, config_space for
    check_and_merge_defaults to be applied to search_options for
    :class:`CostAwareGPFIFOSearcher`.

    :return: (mandatory, default_options, config_space)

    """
    return _common_defaults(is_hyperband=False, is_multi_output=True)


def gp_multifidelity_searcher_defaults() -> (Set[str], dict, dict):
    """
    Returns mandatory, default_options, config_space for
    check_and_merge_defaults to be applied to search_options for
    :class:`GPMultiFidelitySearcher`.

    :return: (mandatory, default_options, config_space)

    """
    return _common_defaults(is_hyperband=True, is_multi_output=False)


def cost_aware_gp_multifidelity_searcher_defaults() -> (Set[str], dict, dict):
    """
    Returns mandatory, default_options, config_space for
    check_and_merge_defaults to be applied to search_options for
    :class:`CostAwareGPMultiFidelitySearcher`.

    :return: (mandatory, default_options, config_space)

    """
    return _common_defaults(is_hyperband=True, is_multi_output=True)
