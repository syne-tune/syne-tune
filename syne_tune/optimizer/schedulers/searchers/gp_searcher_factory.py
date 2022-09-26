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
from typing import Set, Optional
import logging

from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import (
    map_reward_const_minus_x,
    MapReward,
    DEFAULT_INITIAL_SCORING,
    SUPPORTED_INITIAL_SCORING,
    resource_for_acquisition_factory,
    SUPPORTED_RESOURCE_FOR_ACQUISITION,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.kernel_factory import (
    resource_kernel_factory,
    SUPPORTED_RESOURCE_MODELS,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext import (
    ExtendedConfiguration,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    OptimizationConfig,
    DEFAULT_OPTIMIZATION_CONFIG,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_regression import (
    GaussianProcessRegression,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    Matern52,
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
    ScalarMeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.freeze_thaw import (
    ExponentialDecayBaseKernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.model_params import (
    IndependentISSModelParameters,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.gpiss_model import (
    GaussianProcessLearningCurveModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.gpind_model import (
    IndependentGPPerResourceModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.gp_model import (
    HyperTuneIndependentGPModel,
    HyperTuneJointGPModel,
    HyperTuneDistributionArguments,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_skipopt import (
    SkipNoMaxResourcePredicate,
    SkipPeriodicallyPredicate,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import (
    GaussProcEmpiricalBayesModelFactory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gpiss_model import (
    GaussProcAdditiveModelFactory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost_fifo_model import (
    CostSurrogateModelFactory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    EIAcquisitionFunction,
    CEIAcquisitionFunction,
    EIpuAcquisitionFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults import (
    DEFAULT_NUM_INITIAL_CANDIDATES,
    DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
    INTERNAL_CONSTRAINT_NAME,
    INTERNAL_COST_NAME,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    Integer,
    Categorical,
    Boolean,
    Float,
)
from syne_tune.optimizer.schedulers.searchers.utils.warmstarting import (
    create_hp_ranges_for_warmstarting,
    create_filter_observed_data_for_warmstarting,
    create_base_gp_kernel_for_warmstarting,
)
from syne_tune.optimizer.schedulers.searchers.searcher import extract_random_seed

__all__ = [
    "gp_fifo_searcher_factory",
    "gp_multifidelity_searcher_factory",
    "constrained_gp_fifo_searcher_factory",
    "cost_aware_coarse_gp_fifo_searcher_factory",
    "cost_aware_fine_gp_fifo_searcher_factory",
    "cost_aware_gp_multifidelity_searcher_factory",
    "hypertune_searcher_factory",
    "gp_fifo_searcher_defaults",
    "gp_multifidelity_searcher_defaults",
    "constrained_gp_fifo_searcher_defaults",
    "cost_aware_gp_fifo_searcher_defaults",
    "cost_aware_gp_multifidelity_searcher_defaults",
    "hypertune_searcher_defaults",
]

logger = logging.getLogger(__name__)


def _create_base_gp_kernel(hp_ranges: HyperparameterRanges, **kwargs) -> KernelFunction:
    """
    The default base kernel is :class:`Matern52` with ARD parameters.
    But in the transfer learning case, the base kernel is a product of
    two `Matern52` kernels, the first non-ARD over the categorical
    parameter determining the task, the second ARD over the remaining
    parameters.

    """
    if kwargs.get("transfer_learning_task_attr") is not None:
        # Transfer learning: Specific base kernel
        kernel = create_base_gp_kernel_for_warmstarting(hp_ranges, **kwargs)
    else:
        has_covariance_scale = kwargs.get("has_covariance_scale", True)
        kernel = Matern52(
            dimension=hp_ranges.ndarray_size,
            ARD=True,
            has_covariance_scale=has_covariance_scale,
        )
    return kernel


def _create_gp_common(hp_ranges: HyperparameterRanges, **kwargs):
    opt_warmstart = kwargs.get("opt_warmstart", False)
    kernel = _create_base_gp_kernel(hp_ranges, **kwargs)
    mean = ScalarMeanFunction()
    optimization_config = OptimizationConfig(
        lbfgs_tol=DEFAULT_OPTIMIZATION_CONFIG.lbfgs_tol,
        lbfgs_maxiter=kwargs["opt_maxiter"],
        verbose=kwargs["opt_verbose"],
        n_starts=kwargs["opt_nstarts"],
    )
    if kwargs.get("profiler", False):
        profiler = SimpleProfiler()
    else:
        profiler = None
    if kwargs.get("debug_log", False):
        debug_log = DebugLogPrinter()
    else:
        debug_log = None
    filter_observed_data = create_filter_observed_data_for_warmstarting(**kwargs)
    return {
        "opt_warmstart": opt_warmstart,
        "kernel": kernel,
        "mean": mean,
        "optimization_config": optimization_config,
        "profiler": profiler,
        "debug_log": debug_log,
        "filter_observed_data": filter_observed_data,
    }


def _create_gp_model_factory(
    gpmodel,
    result: dict,
    hp_ranges_for_prediction: Optional[HyperparameterRanges],
    active_metric: Optional[str],
    **kwargs,
):
    filter_observed_data = result["filter_observed_data"]
    model_factory = GaussProcEmpiricalBayesModelFactory(
        active_metric=active_metric,
        gpmodel=gpmodel,
        num_fantasy_samples=kwargs["num_fantasy_samples"],
        normalize_targets=kwargs.get("normalize_targets", True),
        profiler=result["profiler"],
        debug_log=result["debug_log"],
        filter_observed_data=filter_observed_data,
        no_fantasizing=kwargs.get("no_fantasizing", False),
        hp_ranges_for_prediction=hp_ranges_for_prediction,
    )
    return {
        "model_factory": model_factory,
        "filter_observed_data": filter_observed_data,
    }


def _create_gp_standard_model(
    hp_ranges: HyperparameterRanges,
    active_metric: Optional[str],
    random_seed: int,
    is_hyperband: bool,
    is_hypertune: bool = False,
    **kwargs,
):
    assert not is_hypertune or is_hyperband
    result = _create_gp_common(hp_ranges, **kwargs)
    kernel = result["kernel"]
    mean = result["mean"]
    if is_hyperband:
        # The `cross-validation` kernel needs an additional argument
        kernel_kwargs = {"num_folds": kwargs["max_epochs"]}
        kernel, mean = resource_kernel_factory(
            kwargs["gp_resource_kernel"], kernel_x=kernel, mean_x=mean, **kernel_kwargs
        )
    common_kwargs = dict(
        kernel=kernel,
        mean=mean,
        optimization_config=result["optimization_config"],
        random_seed=random_seed,
        fit_reset_params=not result["opt_warmstart"],
    )
    if is_hypertune:
        resource_attr_range = (1, kwargs["max_epochs"])
        hypertune_distribution_args = HyperTuneDistributionArguments(
            num_samples=kwargs["hypertune_distribution_num_samples"],
            num_brackets=kwargs["hypertune_distribution_num_brackets"],
        )
        gpmodel = HyperTuneJointGPModel(
            resource_attr_range=resource_attr_range,
            hypertune_distribution_args=hypertune_distribution_args,
            **common_kwargs,
        )
        hp_ranges_for_prediction = hp_ranges
    else:
        gpmodel = GaussianProcessRegression(**common_kwargs)
        hp_ranges_for_prediction = None
    return _create_gp_model_factory(
        gpmodel=gpmodel,
        result=result,
        hp_ranges_for_prediction=hp_ranges_for_prediction,
        active_metric=active_metric,
        **kwargs,
    )


def _create_gp_independent_model(
    hp_ranges: HyperparameterRanges,
    active_metric: Optional[str],
    random_seed: int,
    is_hypertune: bool,
    **kwargs,
):
    def mean_factory(resource: int) -> MeanFunction:
        return ScalarMeanFunction()

    result = _create_gp_common(hp_ranges, has_covariance_scale=False, **kwargs)
    kernel = result["kernel"]
    resource_attr_range = (1, kwargs["max_epochs"])
    common_kwargs = dict(
        kernel=kernel,
        mean_factory=mean_factory,
        resource_attr_range=resource_attr_range,
        optimization_config=result["optimization_config"],
        random_seed=random_seed,
        fit_reset_params=not result["opt_warmstart"],
        separate_noise_variances=kwargs["separate_noise_variances"],
    )
    if is_hypertune:
        hypertune_distribution_args = HyperTuneDistributionArguments(
            num_samples=kwargs["hypertune_distribution_num_samples"],
            num_brackets=kwargs["hypertune_distribution_num_brackets"],
        )
        gpmodel = HyperTuneIndependentGPModel(
            hypertune_distribution_args=hypertune_distribution_args,
            **common_kwargs,
        )
        hp_ranges_for_prediction = hp_ranges
    else:
        gpmodel = IndependentGPPerResourceModel(**common_kwargs)
        hp_ranges_for_prediction = None
    return _create_gp_model_factory(
        gpmodel=gpmodel,
        result=result,
        hp_ranges_for_prediction=hp_ranges_for_prediction,
        active_metric=active_metric,
        **kwargs,
    )


def _create_gp_additive_model(
    model: str,
    hp_ranges: HyperparameterRanges,
    active_metric: Optional[str],
    random_seed: int,
    config_space_ext,
    **kwargs,
):
    result = _create_gp_common(hp_ranges, **kwargs)
    if model == "gp_issm":
        res_model = IndependentISSModelParameters(
            gamma_is_one=kwargs.get("issm_gamma_one", False)
        )
    else:
        assert model == "gp_expdecay", model
        res_model = ExponentialDecayBaseKernelFunction(
            r_max=kwargs["max_epochs"],
            r_min=1,
            normalize_inputs=kwargs.get("expdecay_normalize_inputs", False),
        )
    gpmodel = GaussianProcessLearningCurveModel(
        kernel=result["kernel"],
        res_model=res_model,
        mean=result["mean"],
        optimization_config=result["optimization_config"],
        random_seed=random_seed,
        fit_reset_params=not result["opt_warmstart"],
    )
    filter_observed_data = result["filter_observed_data"]
    no_fantasizing = kwargs.get("no_fantasizing", False)
    num_fantasy_samples = 0 if no_fantasizing else kwargs["num_fantasy_samples"]
    model_factory = GaussProcAdditiveModelFactory(
        gpmodel=gpmodel,
        num_fantasy_samples=num_fantasy_samples,
        active_metric=active_metric,
        config_space_ext=config_space_ext,
        profiler=result["profiler"],
        debug_log=result["debug_log"],
        filter_observed_data=filter_observed_data,
        normalize_targets=kwargs.get("normalize_targets", True),
    )
    return {
        "model_factory": model_factory,
        "filter_observed_data": filter_observed_data,
    }


def _create_common_objects(model=None, is_hypertune=False, **kwargs):
    scheduler = kwargs["scheduler"]
    is_hyperband = scheduler.startswith("hyperband")
    if model is None:
        model = "gp_multitask"
    assert (
        model == "gp_multitask" or is_hyperband
    ), f"model = {model} only together with hyperband_* scheduler"
    hp_ranges = create_hp_ranges_for_warmstarting(**kwargs)
    random_seed, _kwargs = extract_random_seed(kwargs)
    # Skip optimization predicate for GP surrogate model
    if kwargs.get("opt_skip_num_max_resource", False) and is_hyperband:
        skip_optimization = SkipNoMaxResourcePredicate(
            init_length=kwargs["opt_skip_init_length"],
            max_resource=kwargs["max_epochs"],
        )
    elif kwargs.get("opt_skip_period", 1) > 1:
        skip_optimization = SkipPeriodicallyPredicate(
            init_length=kwargs["opt_skip_init_length"], period=kwargs["opt_skip_period"]
        )
    else:
        skip_optimization = None
    # Conversion from reward to metric (strictly decreasing) and back.
    # This is done only if the scheduler mode is 'max'.
    scheduler_mode = kwargs.get("mode", "min")
    if scheduler_mode == "max":
        _map_reward = kwargs.get("map_reward", "1_minus_x")
        if isinstance(_map_reward, str):
            _map_reward_name = _map_reward
            assert _map_reward_name.endswith("minus_x"), (
                f"map_reward = {_map_reward_name} is not supported (use "
                + "'minus_x' or '*_minus_x')"
            )
            if _map_reward_name == "minus_x":
                const = 0.0
            else:
                # Allow strings '*_minus_x', parse const for *
                # Example: '1_minus_x' => const = 1
                offset = len(_map_reward_name) - len("_minus_x")
                const = float(_map_reward_name[:offset])
            _map_reward: Optional[MapReward] = map_reward_const_minus_x(const=const)
        else:
            assert isinstance(
                _map_reward, MapReward
            ), "map_reward must either be string or of MapReward type"
    else:
        assert (
            scheduler_mode == "min"
        ), f"mode = {scheduler_mode}, must be in ('max', 'min')"
        _map_reward = kwargs.get("map_reward")
        if _map_reward is not None:
            logger.warning(
                f"Since mode == 'min', map_reward = {_map_reward} is ignored"
            )
            _map_reward = None
    result = {
        "hp_ranges": hp_ranges,
        "map_reward": _map_reward,
        "skip_optimization": skip_optimization,
    }
    if is_hyperband:
        epoch_range = (1, kwargs["max_epochs"])
        result["config_space_ext"] = ExtendedConfiguration(
            hp_ranges,
            resource_attr_key=kwargs["resource_attr"],
            resource_attr_range=epoch_range,
        )

    # Create model factory
    if model == "gp_multitask":
        result.update(
            _create_gp_standard_model(
                hp_ranges=hp_ranges,
                active_metric=INTERNAL_METRIC_NAME,
                random_seed=random_seed,
                is_hyperband=is_hyperband,
                is_hypertune=is_hypertune,
                **_kwargs,
            )
        )
    elif model == "gp_independent":
        result.update(
            _create_gp_independent_model(
                hp_ranges=hp_ranges,
                active_metric=INTERNAL_METRIC_NAME,
                random_seed=random_seed,
                is_hypertune=is_hypertune,
                **_kwargs,
            )
        )
    elif model == "gp_independent":
        result.update(
            _create_gp_independent_model(
                hp_ranges=hp_ranges,
                active_metric=INTERNAL_METRIC_NAME,
                random_seed=random_seed,
                **_kwargs,
            )
        )
    else:
        result.update(
            _create_gp_additive_model(
                model=model,
                hp_ranges=hp_ranges,
                active_metric=INTERNAL_METRIC_NAME,
                random_seed=random_seed,
                config_space_ext=result["config_space_ext"],
                **_kwargs,
            )
        )
    result["num_initial_candidates"] = kwargs["num_init_candidates"]
    result["num_initial_random_choices"] = kwargs["num_init_random"]
    for k in ("initial_scoring", "cost_attr", "skip_local_optimization"):
        result[k] = kwargs[k]

    return result


def gp_fifo_searcher_factory(**kwargs) -> dict:
    """
    Returns kwargs for `GPFIFOSearcher._create_internal`, based on kwargs
    equal to search_options passed to and extended by scheduler (see
    :class:`FIFOScheduler`).

    Extensions of kwargs by the scheduler:
    - scheduler: Name of scheduler ('fifo', 'hyperband_*')
    - config_space: Configuration space
    Only Hyperband schedulers:
    - resource_attr: Name of resource (or time) attribute
    - max_epochs: Maximum resource value

    :param kwargs: search_options coming from scheduler
    :return: kwargs for GPFIFOSearcher._create_internal

    """
    assert (
        kwargs["scheduler"] == "fifo"
    ), "This factory needs scheduler = 'fifo' (instead of '{}')".format(
        kwargs["scheduler"]
    )
    # Common objects
    result = _create_common_objects(**kwargs)

    return dict(**result, acquisition_class=EIAcquisitionFunction)


def gp_multifidelity_searcher_factory(**kwargs) -> dict:
    """
    Returns kwargs for `GPMultiFidelitySearcher._create_internal`, based on
    kwargs equal to search_options passed to and extended by scheduler (see
    :class:`HyperbandScheduler`).

    :param kwargs: search_options coming from scheduler
    :return: kwargs for GPMultiFidelitySearcher._create_internal

    """
    supp_schedulers = {
        "hyperband_stopping",
        "hyperband_promotion",
        "hyperband_synchronous",
        "hyperband_pasha",
    }
    assert (
        kwargs["scheduler"] in supp_schedulers
    ), "This factory needs scheduler in {} (instead of '{}')".format(
        supp_schedulers, kwargs["scheduler"]
    )
    if kwargs.get("model") is None:
        kwargs["model"] = "gp_multitask"
    # Common objects
    result = _create_common_objects(**kwargs)
    kwargs_int = dict(
        result,
        resource_attr=kwargs["resource_attr"],
        acquisition_class=EIAcquisitionFunction,
    )
    if kwargs["model"] in {"gp_multitask", "gp_independent"}:
        kwargs_int["resource_for_acquisition"] = resource_for_acquisition_factory(
            kwargs, result["hp_ranges"]
        )
    return kwargs_int


def hypertune_searcher_factory(**kwargs) -> dict:
    if kwargs.get("model") is None:
        kwargs["model"] = "gp_independent"
    else:
        supported_models = {"gp_multitask", "gp_independent"}
        assert kwargs["model"] in supported_models, (
            "Hyper-Tune only supports search_options['model'] in "
            f"{supported_models} along with searcher = 'hypertune'"
        )
    return gp_multifidelity_searcher_factory(**kwargs, is_hypertune=True)


def constrained_gp_fifo_searcher_factory(**kwargs) -> dict:
    """
    Returns kwargs for `ConstrainedGPFIFOSearcher._create_internal`, based on kwargs
    equal to search_options passed to and extended by scheduler (see
    :class:`FIFOScheduler`).

    :param kwargs: search_options coming from scheduler
    :return: kwargs for ConstrainedGPFIFOSearcher._create_internal

    """
    assert (
        kwargs["scheduler"] == "fifo"
    ), "This factory needs scheduler = 'fifo' (instead of '{}')".format(
        kwargs["scheduler"]
    )
    # Common objects
    result = _create_common_objects(**kwargs)
    model_factory = result.pop("model_factory")
    skip_optimization = result.pop("skip_optimization")
    # We need two model factories: one for active metric (model_factory),
    # the other for constraint metric (model_factory_constraint)
    random_seed, _kwargs = extract_random_seed(kwargs)
    model_factory_constraint = _create_gp_standard_model(
        hp_ranges=result["hp_ranges"],
        active_metric=INTERNAL_CONSTRAINT_NAME,
        random_seed=random_seed,
        is_hyperband=False,
        **_kwargs,
    )["model_factory"]
    # Sharing debug_log attribute across models
    model_factory_constraint._debug_log = model_factory._debug_log
    # The same skip_optimization strategy applies to both models
    skip_optimization_constraint = skip_optimization

    output_model_factory = {
        INTERNAL_METRIC_NAME: model_factory,
        INTERNAL_CONSTRAINT_NAME: model_factory_constraint,
    }
    output_skip_optimization = {
        INTERNAL_METRIC_NAME: skip_optimization,
        INTERNAL_CONSTRAINT_NAME: skip_optimization_constraint,
    }

    return dict(
        result,
        output_model_factory=output_model_factory,
        output_skip_optimization=output_skip_optimization,
        acquisition_class=CEIAcquisitionFunction,
    )


def cost_aware_coarse_gp_fifo_searcher_factory(**kwargs) -> dict:
    """
    Returns kwargs for `CostAwareGPFIFOSearcher._create_internal`, based on
    kwargs equal to search_options passed to and extended by scheduler (see
    :class:`FIFOScheduler`).
    This is for the coarse-grained variant, where costs c(x) are obtained
    together with metric values and are given a GP surrogate model.

    :param kwargs: search_options coming from scheduler
    :return: kwargs for CostAwareGPFIFOSearcher._create_internal

    """
    assert (
        kwargs["scheduler"] == "fifo"
    ), "This factory needs scheduler = 'fifo' (instead of '{}')".format(
        kwargs["scheduler"]
    )
    # Common objects
    result = _create_common_objects(**kwargs)
    model_factory = result.pop("model_factory")
    skip_optimization = result.pop("skip_optimization")
    # We need two model factories: one for active metric (model_factory),
    # the other for cost metric (model_factory_cost)
    random_seed, _kwargs = extract_random_seed(kwargs)
    model_factory_cost = _create_gp_standard_model(
        hp_ranges=result["hp_ranges"],
        active_metric=INTERNAL_COST_NAME,
        random_seed=random_seed,
        is_hyperband=False,
        **_kwargs,
    )["model_factory"]
    # Sharing debug_log attribute across models
    model_factory_cost._debug_log = model_factory._debug_log
    exponent_cost = kwargs.get("exponent_cost", 1.0)
    acquisition_class = (EIpuAcquisitionFunction, dict(exponent_cost=exponent_cost))
    # The same skip_optimization strategy applies to both models
    skip_optimization_cost = skip_optimization

    output_model_factory = {
        INTERNAL_METRIC_NAME: model_factory,
        INTERNAL_COST_NAME: model_factory_cost,
    }
    output_skip_optimization = {
        INTERNAL_METRIC_NAME: skip_optimization,
        INTERNAL_COST_NAME: skip_optimization_cost,
    }

    return dict(
        result,
        output_model_factory=output_model_factory,
        output_skip_optimization=output_skip_optimization,
        acquisition_class=acquisition_class,
    )


def cost_aware_fine_gp_fifo_searcher_factory(**kwargs) -> dict:
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
    assert kwargs["scheduler"] in [
        "fifo"
    ], "This factory needs scheduler = 'fifo' (instead of '{}')".format(
        kwargs["scheduler"]
    )
    cost_model = kwargs.get("cost_model")
    assert cost_model is not None, (
        "If search_options['resource_attr'] is given, a CostModel has "
        + "to be specified in search_options['cost_model']"
    )
    fixed_resource = kwargs.get("max_epochs")
    assert fixed_resource is not None, (
        "If search_options['resource_attr'] is given, the maximum "
        + "resource level has to be specified in "
        + "search_options['max_epochs'], or (simpler) as max_t when "
        + "creating FIFOScheduler"
    )
    # Common objects
    result = _create_common_objects(**kwargs)
    model_factory = result.pop("model_factory")
    skip_optimization = result.pop("skip_optimization")
    # We need two model factories: one for active metric (model_factory),
    # the other for cost metric (model_factory_cost)
    model_factory_cost = CostSurrogateModelFactory(
        model=kwargs["cost_model"], fixed_resource=fixed_resource, num_samples=1
    )
    exponent_cost = kwargs.get("exponent_cost", 1.0)
    acquisition_class = (EIpuAcquisitionFunction, dict(exponent_cost=exponent_cost))
    # The same skip_optimization strategy applies to both models
    skip_optimization_cost = skip_optimization

    output_model_factory = {
        INTERNAL_METRIC_NAME: model_factory,
        INTERNAL_COST_NAME: model_factory_cost,
    }
    output_skip_optimization = {
        INTERNAL_METRIC_NAME: skip_optimization,
        INTERNAL_COST_NAME: skip_optimization_cost,
    }

    return dict(
        result,
        output_model_factory=output_model_factory,
        output_skip_optimization=output_skip_optimization,
        acquisition_class=acquisition_class,
        resource_attr=kwargs["resource_attr"],
    )


def cost_aware_gp_multifidelity_searcher_factory(**kwargs) -> dict:
    """
    Returns kwargs for `CostAwareGPMultiFidelitySearcher._create_internal`,
    based on kwargs equal to search_options passed to and extended by
    scheduler (see :class:`HyperbandScheduler`).

    :param kwargs: search_options coming from scheduler
    :return: kwargs for CostAwareGPMultiFidelitySearcher._create_internal

    """
    supp_schedulers = {
        "hyperband_stopping",
        "hyperband_promotion",
        "hyperband_synchronous",
        "hyperband_pasha",
    }
    assert (
        kwargs["scheduler"] in supp_schedulers
    ), "This factory needs scheduler in {} (instead of '{}')".format(
        supp_schedulers, kwargs["scheduler"]
    )
    cost_model = kwargs.get("cost_model")
    assert cost_model is not None, (
        "If search_options['resource_attr'] is given, a CostModel has "
        + "to be specified in search_options['cost_model']"
    )
    # Common objects
    result = _create_common_objects(**kwargs)
    model_factory = result.pop("model_factory")
    skip_optimization = result.pop("skip_optimization")
    # We need two model factories: one for active metric (model_factory),
    # the other for cost metric (model_factory_cost)
    model_factory_cost = CostSurrogateModelFactory(
        model=kwargs["cost_model"], fixed_resource=kwargs["max_epochs"], num_samples=1
    )
    exponent_cost = kwargs.get("exponent_cost", 1.0)
    acquisition_class = (EIpuAcquisitionFunction, dict(exponent_cost=exponent_cost))
    # The same skip_optimization strategy applies to both models
    skip_optimization_cost = skip_optimization

    output_model_factory = {
        INTERNAL_METRIC_NAME: model_factory,
        INTERNAL_COST_NAME: model_factory_cost,
    }
    output_skip_optimization = {
        INTERNAL_METRIC_NAME: skip_optimization,
        INTERNAL_COST_NAME: skip_optimization_cost,
    }

    resource_for_acquisition = resource_for_acquisition_factory(
        kwargs, result["hp_ranges"]
    )
    return dict(
        result,
        resource_attr=kwargs["resource_attr"],
        output_model_factory=output_model_factory,
        output_skip_optimization=output_skip_optimization,
        resource_for_acquisition=resource_for_acquisition,
        acquisition_class=acquisition_class,
    )


def _common_defaults(
    is_hyperband: bool, is_multi_output: bool, is_hypertune: bool = False
) -> (Set[str], dict, dict):
    mandatory = set()

    default_options = {
        "opt_skip_init_length": 150,
        "opt_skip_period": 1,
        "profiler": False,
        "opt_maxiter": 50,
        "opt_nstarts": 2,
        "opt_warmstart": False,
        "opt_verbose": False,
        "opt_debug_writer": False,
        "num_fantasy_samples": 20,
        "scheduler": "fifo",
        "num_init_random": DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
        "num_init_candidates": DEFAULT_NUM_INITIAL_CANDIDATES,
        "initial_scoring": DEFAULT_INITIAL_SCORING,
        "skip_local_optimization": False,
        "debug_log": True,
        "cost_attr": "elapsed_time",
        "normalize_targets": True,
        "no_fantasizing": False,
    }
    if is_hyperband:
        if is_hypertune:
            default_options["model"] = "gp_independent"
        else:
            default_options["model"] = "gp_multitask"
        default_options["opt_skip_num_max_resource"] = False
        default_options["gp_resource_kernel"] = "exp-decay-sum"
        default_options["resource_acq"] = "bohb"
        default_options["resource_acq_bohb_threshold"] = 3
        default_options["num_init_random"] = 6
        default_options["issm_gamma_one"] = False
        default_options["expdecay_normalize_inputs"] = False
        default_options["separate_noise_variances"] = False
        default_options["hypertune_distribution_num_samples"] = 50
        default_options["hypertune_distribution_num_brackets"] = 1
    if is_multi_output:
        default_options["initial_scoring"] = "acq_func"
        default_options["exponent_cost"] = 1.0

    constraints = {
        "random_seed": Integer(0, 2**32 - 1),
        "opt_skip_init_length": Integer(0, None),
        "opt_skip_period": Integer(1, None),
        "profiler": Boolean(),
        "opt_maxiter": Integer(1, None),
        "opt_nstarts": Integer(1, None),
        "opt_warmstart": Boolean(),
        "opt_verbose": Boolean(),
        "opt_debug_writer": Boolean(),
        "num_fantasy_samples": Integer(1, None),
        "num_init_random": Integer(0, None),
        "num_init_candidates": Integer(5, None),
        "initial_scoring": Categorical(choices=tuple(SUPPORTED_INITIAL_SCORING)),
        "skip_local_optimization": Boolean(),
        "debug_log": Boolean(),
        "normalize_targets": Boolean(),
    }

    if is_hyperband:
        model_choices = ("gp_multitask", "gp_independent")
        if not is_hypertune:
            model_choices = model_choices + ("gp_issm", "gp_expdecay")
        constraints["model"] = Categorical(choices=model_choices)
        constraints["opt_skip_num_max_resource"] = Boolean()
        constraints["gp_resource_kernel"] = Categorical(
            choices=SUPPORTED_RESOURCE_MODELS
        )
        constraints["resource_acq"] = Categorical(
            choices=tuple(SUPPORTED_RESOURCE_FOR_ACQUISITION)
        )
        constraints["issm_gamma_one"] = Boolean()
        constraints["expdecay_normalize_inputs"] = Boolean()
        constraints["separate_noise_variances"] = Boolean()
        constraints["hypertune_distribution_num_samples"] = Integer(1, None)
        constraints["hypertune_distribution_num_brackets"] = Integer(1, None)
    if is_multi_output:
        constraints["initial_scoring"] = Categorical(choices=tuple({"acq_func"}))
        constraints["exponent_cost"] = Float(0.0, 1.0)

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


def hypertune_searcher_defaults() -> (Set[str], dict, dict):
    """
    Returns mandatory, default_options, config_space for
    check_and_merge_defaults to be applied to search_options for
    :class:`HyperTuneSearcher`.

    :return: (mandatory, default_options, config_space)

    """
    return _common_defaults(is_hyperband=True, is_multi_output=False, is_hypertune=True)


def cost_aware_gp_multifidelity_searcher_defaults() -> (Set[str], dict, dict):
    """
    Returns mandatory, default_options, config_space for
    check_and_merge_defaults to be applied to search_options for
    :class:`CostAwareGPMultiFidelitySearcher`.

    :return: (mandatory, default_options, config_space)

    """
    return _common_defaults(is_hyperband=True, is_multi_output=True)
