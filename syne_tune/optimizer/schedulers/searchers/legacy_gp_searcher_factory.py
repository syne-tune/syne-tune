from typing import Set, Optional, Dict, Any
import logging
from functools import partial

from syne_tune.optimizer.schedulers.searchers.legacy_gp_searcher_utils import (
    map_reward_const_minus_x,
    MapReward,
    DEFAULT_INITIAL_SCORING,
    SUPPORTED_INITIAL_SCORING,
    resource_for_acquisition_factory,
    SUPPORTED_RESOURCE_FOR_ACQUISITION,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.kernel_factory import (
    SUPPORTED_BASE_MODELS,
    base_kernel_factory,
    SUPPORTED_RESOURCE_MODELS,
    resource_kernel_factory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.acqfunc_factory import (
    acquisition_function_factory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext import (
    ExtendedConfiguration,
)
from syne_tune.optimizer.schedulers.searchers.utils import (
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
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.warping import (
    kernel_with_warping,
    WarpedKernel,
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
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.target_transform import (
    BoxCoxTargetTransform,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_skipopt import (
    SkipNoMaxResourcePredicate,
    SkipPeriodicallyPredicate,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import (
    GaussProcEmpiricalBayesEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gpiss_model import (
    GaussProcAdditiveEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost_fifo_model import (
    CostEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    CEIAcquisitionFunction,
    EIpuAcquisitionFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults import (
    DEFAULT_NUM_INITIAL_CANDIDATES,
    DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
    DEFAULT_MAX_SIZE_DATA_FOR_MODEL,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
    INTERNAL_CONSTRAINT_NAME,
    INTERNAL_COST_NAME,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.subsample_state_multi_fidelity import (
    SubsampleMultiFidelityStateConverter,
    SubsampleMFDenseDataStateConverter,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.subsample_state_single_fidelity import (
    SubsampleSingleFidelityStateConverter,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    StateForModelConverter,
)
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    Integer,
    Categorical,
    Boolean,
    Float,
    IntegerOrNone,
)
from syne_tune.optimizer.schedulers.searchers.utils.warmstarting import (
    create_hp_ranges_for_warmstarting,
    create_filter_observed_data_for_warmstarting,
    create_base_gp_kernel_for_warmstarting,
)
from syne_tune.optimizer.schedulers.searchers import extract_random_seed
from syne_tune.optimizer.schedulers.random_seeds import RANDOM_SEED_UPPER_BOUND


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
    two ``Matern52`` kernels, the first non-ARD over the categorical
    parameter determining the task, the second ARD over the remaining
    parameters.
    """
    input_warping = kwargs.get("input_warping", False)
    if kwargs.get("transfer_learning_task_attr") is not None:
        if input_warping:
            logger.warning(
                "Cannot use input_warping=True together with transfer_learning_task_attr. Will use input_warping=False"
            )
        # Transfer learning: Specific base kernel
        kernel = create_base_gp_kernel_for_warmstarting(hp_ranges, **kwargs)
    else:
        has_covariance_scale = kwargs.get("has_covariance_scale", True)
        kernel = base_kernel_factory(
            name=kwargs["gp_base_kernel"],
            dimension=hp_ranges.ndarray_size,
            has_covariance_scale=has_covariance_scale,
        )
        if input_warping:
            # Use input warping on all coordinates which do not belong to a
            # categorical hyperparameter
            kernel = kernel_with_warping(kernel, hp_ranges)
            if kwargs.get("debug_log", False) and isinstance(kernel, WarpedKernel):
                ranges = [(warp.lower, warp.upper) for warp in kernel.warpings]
                logger.info(
                    f"Creating base GP covariance kernel with input warping: ranges = {ranges}"
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
    if kwargs.get("debug_log", False):
        debug_log = DebugLogPrinter()
    else:
        debug_log = None
    if kwargs.get("boxcox_transform", False):
        target_transform = BoxCoxTargetTransform()
    else:
        target_transform = None
    filter_observed_data = create_filter_observed_data_for_warmstarting(**kwargs)
    return {
        "opt_warmstart": opt_warmstart,
        "kernel": kernel,
        "mean": mean,
        "target_transform": target_transform,
        "optimization_config": optimization_config,
        "debug_log": debug_log,
        "filter_observed_data": filter_observed_data,
    }


def _create_gp_estimator(
    gpmodel,
    result: Dict[str, Any],
    hp_ranges_for_prediction: Optional[HyperparameterRanges],
    active_metric: Optional[str],
    **kwargs,
):
    filter_observed_data = result["filter_observed_data"]
    estimator = GaussProcEmpiricalBayesEstimator(
        active_metric=active_metric,
        gpmodel=gpmodel,
        num_fantasy_samples=kwargs["num_fantasy_samples"],
        normalize_targets=kwargs.get("normalize_targets", True),
        debug_log=result["debug_log"],
        filter_observed_data=filter_observed_data,
        no_fantasizing=kwargs.get("no_fantasizing", False),
        hp_ranges_for_prediction=hp_ranges_for_prediction,
    )
    return {
        "estimator": estimator,
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
        # The ``cross-validation`` kernel needs an additional argument
        kernel_kwargs = {"num_folds": kwargs["max_epochs"]}
        kernel, mean = resource_kernel_factory(
            kwargs["gp_resource_kernel"], kernel_x=kernel, mean_x=mean, **kernel_kwargs
        )
    common_kwargs = dict(
        kernel=kernel,
        mean=mean,
        target_transform=result["target_transform"],
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
    return _create_gp_estimator(
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
        target_transform=result["target_transform"],
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
    return _create_gp_estimator(
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
    estimator = GaussProcAdditiveEstimator(
        gpmodel=gpmodel,
        num_fantasy_samples=num_fantasy_samples,
        active_metric=active_metric,
        config_space_ext=config_space_ext,
        debug_log=result["debug_log"],
        filter_observed_data=filter_observed_data,
        normalize_targets=kwargs.get("normalize_targets", True),
    )
    return {
        "estimator": estimator,
        "filter_observed_data": filter_observed_data,
    }


def _create_state_converter(
    model: str,
    is_hyperband: bool,
    **kwargs,
) -> Optional[StateForModelConverter]:
    """
    For model-based multi-fidelity methods, if ``max_size_data_for_model`` is given,
    we use a state converter which reduces the number of observed datapoints to
    ``max_size_data_for_model``. There are different such converters, depending on
    which method is being used.

    Note: These state converters need a ``random_state``. This is not created here,
    but is assigned later, in order to maintain control of random seeds

    :param kwargs: Arguments
    :return: State converter; or ``None`` if ``max_size_data_for_model`` not given
    """
    max_size = kwargs.get("max_size_data_for_model")
    if max_size is None:
        return None
    if is_hyperband:
        if model not in ("gp_multitask", "gp_independent"):
            logger.warning(
                f"Cannot use max_size_data_for_model together with model={model}"
            )
            return None
        if kwargs.get("searcher_data") == "all":
            logger.warning(
                f"You are using max_size_data_for_model={max_size} together with "
                f"model={model} and searcher_data='all'. This may lead to poor "
                "results. Use searcher_data='rungs' to limit the size of the data, "
                "which you can combine with max_size_data_for_model"
            )
        if kwargs["scheduler"] == "hyperband_dyhpo":
            # In DyHPO, there is too much data for some trials, so we need to limit the
            # data differently
            # Note: We use the defaults for ``grace_period`` and ``reduction_factor``
            # here, could make them configurable as well.
            return SubsampleMFDenseDataStateConverter(max_size=max_size)
        else:
            return SubsampleMultiFidelityStateConverter(max_size=max_size)
    else:
        scheduler_mode = kwargs.get("mode", "min")
        top_fraction = kwargs["max_size_top_fraction"]
        return SubsampleSingleFidelityStateConverter(
            max_size=max_size, mode=scheduler_mode, top_fraction=top_fraction
        )


def _create_common_objects(model=None, is_hypertune=False, **kwargs):
    scheduler = kwargs["scheduler"]
    is_hyperband = scheduler.startswith("hyperband")
    if model is None:
        model = "gp_multitask"
    assert (
        model == "gp_multitask" or is_hyperband
    ), f"model = {model} only together with hyperband_* scheduler"
    hp_ranges = create_hp_ranges_for_warmstarting(**kwargs)
    random_seed, _kwargs = extract_random_seed(**kwargs)
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
        # Extended config space
        epoch_range = (1, kwargs["max_epochs"])
        result["config_space_ext"] = ExtendedConfiguration(
            hp_ranges,
            resource_attr_key=kwargs["resource_attr"],
            resource_attr_range=epoch_range,
        )
    # State converter to down sample data
    state_converter = _create_state_converter(model, is_hyperband, **kwargs)
    if state_converter is not None:
        result["state_converter"] = state_converter

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
    for k in (
        "initial_scoring",
        "cost_attr",
        "skip_local_optimization",
        "allow_duplicates",
    ):
        result[k] = kwargs[k]
    if kwargs.get("restrict_configurations") is not None:
        # If ``restrict_configurations`` is given, the searcher may only suggest
        # one the configs in this list. This rules out local optimization of the
        # acquisition function
        result["skip_local_optimization"] = True

    return result


def _create_acq_function(**kwargs):
    name = kwargs["acq_function"]
    af_kwargs = kwargs.get("acq_function_kwargs", dict())
    return acquisition_function_factory(name, **af_kwargs)


def gp_fifo_searcher_factory(**kwargs) -> Dict[str, Any]:
    """
    Returns ``kwargs`` for
    :meth:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher._create_internal`,
    based on ``kwargs`` equal to ``search_options`` passed to and extended by
    scheduler (see :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`).

    Extensions of ``kwargs`` by the scheduler:

    * ``scheduler``: Name of scheduler ``("fifo", "hyperband_*")``
    * ``config_space``: Configuration space

    Only Hyperband schedulers:

    * ``resource_attr``: Name of resource (or time) attribute
    * ``max_epochs``: Maximum resource value

    :param kwargs: ``search_options`` coming from scheduler
    :return: ``kwargs`` for :meth:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher._create_internal`
    """
    assert (
        kwargs["scheduler"] == "fifo"
    ), "This factory needs scheduler = 'fifo' (instead of '{}')".format(
        kwargs["scheduler"]
    )
    # Common objects
    result = _create_common_objects(**kwargs)

    return dict(**result, acquisition_class=_create_acq_function(**kwargs))


def gp_multifidelity_searcher_factory(**kwargs) -> Dict[str, Any]:
    """
    Returns ``kwargs`` for
    :meth:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher._create_internal`,
    based on ``kwargs`` equal to ``search_options`` passed to and extended by
    scheduler (see :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`).

    :param kwargs: ``search_options`` coming from scheduler
    :return: ``kwargs`` for :meth:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher._create_internal`
    """
    supp_schedulers = {
        "hyperband_stopping",
        "hyperband_promotion",
        "hyperband_synchronous",
        "hyperband_pasha",
        "hyperband_dyhpo",
        "hyperband_legacy_dyhpo",
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
        acquisition_class=_create_acq_function(**kwargs),
    )
    if kwargs["model"] in {"gp_multitask", "gp_independent"}:
        kwargs_int["resource_for_acquisition"] = resource_for_acquisition_factory(
            kwargs, result["hp_ranges"]
        )
    return kwargs_int


def hypertune_searcher_factory(**kwargs) -> Dict[str, Any]:
    """
    Returns ``kwargs`` for
    :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_hypertune.HyperTuneSearcher._create_internal`,
    based on ``kwargs`` equal to ``search_options`` passed to and extended by
    scheduler (see :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`).

    :param kwargs: ``search_options`` coming from scheduler
    :return: ``kwargs`` for :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_hypertune.HyperTuneSearcher._create_internal`
    """
    if kwargs.get("model") is None:
        kwargs["model"] = "gp_independent"
    else:
        supported_models = {"gp_multitask", "gp_independent"}
        assert kwargs["model"] in supported_models, (
            "Hyper-Tune only supports search_options['model'] in "
            f"{supported_models} along with searcher = 'legacy_hypertune'"
        )
    return gp_multifidelity_searcher_factory(**kwargs, is_hypertune=True)


def constrained_gp_fifo_searcher_factory(**kwargs) -> Dict[str, Any]:
    """
    Returns ``kwargs`` for
    :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_constrained.ConstrainedGPFIFOSearcher._create_internal`,
    based on ``kwargs`` equal to ``search_options`` passed to and extended by
    scheduler (see :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`).

    :param kwargs: ``search_options`` coming from scheduler
    :return: ``kwargs`` for :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_constrained.ConstrainedGPFIFOSearcher._create_internal`
    """
    assert (
        kwargs["scheduler"] == "fifo"
    ), "This factory needs scheduler = 'fifo' (instead of '{}')".format(
        kwargs["scheduler"]
    )
    # Common objects
    result = _create_common_objects(**kwargs)
    estimator = result.pop("estimator")
    skip_optimization = result.pop("skip_optimization")
    # We need two model factories: one for active metric (estimator),
    # the other for constraint metric (estimator_constraint)
    random_seed, _kwargs = extract_random_seed(**kwargs)
    estimator_constraint = _create_gp_standard_model(
        hp_ranges=result["hp_ranges"],
        active_metric=INTERNAL_CONSTRAINT_NAME,
        random_seed=random_seed,
        is_hyperband=False,
        **_kwargs,
    )["estimator"]
    # Sharing debug_log attribute across models
    estimator_constraint._debug_log = estimator._debug_log
    # The same skip_optimization strategy applies to both models
    skip_optimization_constraint = skip_optimization

    output_estimator = {
        INTERNAL_METRIC_NAME: estimator,
        INTERNAL_CONSTRAINT_NAME: estimator_constraint,
    }
    output_skip_optimization = {
        INTERNAL_METRIC_NAME: skip_optimization,
        INTERNAL_CONSTRAINT_NAME: skip_optimization_constraint,
    }

    return dict(
        result,
        output_estimator=output_estimator,
        output_skip_optimization=output_skip_optimization,
        acquisition_class=CEIAcquisitionFunction,
    )


def cost_aware_coarse_gp_fifo_searcher_factory(**kwargs) -> Dict[str, Any]:
    """
    Returns ``kwargs`` for
    :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.CostAwareGPFIFOSearcher._create_internal`,
    based on ``kwargs`` equal to ``search_options`` passed to and extended by
    scheduler (see :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`).

    This is for the coarse-grained variant, where costs :math:`c(x)` are obtained
    together with metric values and are given a GP surrogate model.

    :param kwargs: ``search_options`` coming from scheduler
    :return: ``kwargs`` for :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.CostAwareGPFIFOSearcher._create_internal`
    """
    assert (
        kwargs["scheduler"] == "fifo"
    ), "This factory needs scheduler = 'fifo' (instead of '{}')".format(
        kwargs["scheduler"]
    )
    # Common objects
    result = _create_common_objects(**kwargs)
    estimator = result.pop("estimator")
    skip_optimization = result.pop("skip_optimization")
    # We need two model factories: one for active metric (estimator),
    # the other for cost metric (estimator_cost)
    random_seed, _kwargs = extract_random_seed(**kwargs)
    estimator_cost = _create_gp_standard_model(
        hp_ranges=result["hp_ranges"],
        active_metric=INTERNAL_COST_NAME,
        random_seed=random_seed,
        is_hyperband=False,
        **_kwargs,
    )["estimator"]
    # Sharing debug_log attribute across models
    estimator_cost._debug_log = estimator._debug_log
    exponent_cost = kwargs.get("exponent_cost", 1.0)
    acquisition_class = partial(EIpuAcquisitionFunction, exponent_cost=exponent_cost)
    # The same skip_optimization strategy applies to both models
    skip_optimization_cost = skip_optimization

    output_estimator = {
        INTERNAL_METRIC_NAME: estimator,
        INTERNAL_COST_NAME: estimator_cost,
    }
    output_skip_optimization = {
        INTERNAL_METRIC_NAME: skip_optimization,
        INTERNAL_COST_NAME: skip_optimization_cost,
    }

    return dict(
        result,
        output_estimator=output_estimator,
        output_skip_optimization=output_skip_optimization,
        acquisition_class=acquisition_class,
    )


def cost_aware_fine_gp_fifo_searcher_factory(**kwargs) -> Dict[str, Any]:
    """
    Returns ``kwargs`` for
    :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.CostAwareGPFIFOSearcher._create_internal`,
    based on ``kwargs`` equal to ``search_options`` passed to and extended by
    scheduler (see :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`).

    This is for the fine-grained variant, where costs :math:`c(x, r)` are
    obtained with each report and are represented by a
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.cost_model.CostModel`
    surrogate model.

    :param kwargs: ``search_options`` coming from scheduler
    :return: ``kwargs`` for :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.CostAwareGPFIFOSearcher._create_internal`
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
    estimator = result.pop("estimator")
    skip_optimization = result.pop("skip_optimization")
    # We need two model factories: one for active metric (estimator),
    # the other for cost metric (estimator_cost)
    estimator_cost = CostEstimator(
        model=kwargs["cost_model"], fixed_resource=fixed_resource, num_samples=1
    )
    exponent_cost = kwargs.get("exponent_cost", 1.0)
    acquisition_class = partial(EIpuAcquisitionFunction, exponent_cost=exponent_cost)
    # The same skip_optimization strategy applies to both models
    skip_optimization_cost = skip_optimization

    output_estimator = {
        INTERNAL_METRIC_NAME: estimator,
        INTERNAL_COST_NAME: estimator_cost,
    }
    output_skip_optimization = {
        INTERNAL_METRIC_NAME: skip_optimization,
        INTERNAL_COST_NAME: skip_optimization_cost,
    }

    return dict(
        result,
        output_estimator=output_estimator,
        output_skip_optimization=output_skip_optimization,
        acquisition_class=acquisition_class,
        resource_attr=kwargs["resource_attr"],
    )


def cost_aware_gp_multifidelity_searcher_factory(**kwargs) -> Dict[str, Any]:
    """
    Returns ``kwargs`` for
    :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.CostAwareGPMultiFidelitySearcher._create_internal`,
    based on ``kwargs`` equal to ``search_options`` passed to and extended by
    scheduler (see :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`).

    :param kwargs: ``search_options`` coming from scheduler
    :return: ``kwargs`` for :meth:`~syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.CostAwareGPMultiFidelitySearcher._create_internal`
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
    estimator = result.pop("estimator")
    skip_optimization = result.pop("skip_optimization")
    # We need two model factories: one for active metric (estimator),
    # the other for cost metric (estimator_cost)
    estimator_cost = CostEstimator(
        model=kwargs["cost_model"], fixed_resource=kwargs["max_epochs"], num_samples=1
    )
    exponent_cost = kwargs.get("exponent_cost", 1.0)
    acquisition_class = partial(EIpuAcquisitionFunction, exponent_cost=exponent_cost)
    # The same skip_optimization strategy applies to both models
    skip_optimization_cost = skip_optimization

    output_estimator = {
        INTERNAL_METRIC_NAME: estimator,
        INTERNAL_COST_NAME: estimator_cost,
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
        output_estimator=output_estimator,
        output_skip_optimization=output_skip_optimization,
        resource_for_acquisition=resource_for_acquisition,
        acquisition_class=acquisition_class,
    )


def _common_defaults(
    kwargs: Dict[str, Any],
    is_hyperband: bool,
    is_multi_output: bool = False,
    is_hypertune: bool = False,
    is_restrict_configs: bool = False,
) -> (Set[str], dict, dict):
    mandatory = set()

    default_options = {
        "opt_skip_init_length": 150,
        "opt_skip_period": 1,
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
        "debug_log": False,
        "cost_attr": "elapsed_time",
        "normalize_targets": True,
        "no_fantasizing": False,
        "allow_duplicates": False,
        "input_warping": False,
        "boxcox_transform": False,
        "max_size_top_fraction": 0.25,
        "gp_base_kernel": "matern52-ard",
        "acq_function": "ei",
    }
    if is_restrict_configs:
        default_options["initial_scoring"] = "acq_func"
        default_options["skip_local_optimization"] = True
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
    if (not is_hyperband) or (
        kwargs.get("model") in (None, "gp_multitask", "gp_independent")
        and kwargs.get("searcher_data") != "all"
    ):
        default_options["max_size_data_for_model"] = DEFAULT_MAX_SIZE_DATA_FOR_MODEL
    if is_multi_output:
        default_options["initial_scoring"] = "acq_func"
        default_options["exponent_cost"] = 1.0

    constraints = {
        "random_seed": Integer(0, RANDOM_SEED_UPPER_BOUND),
        "opt_skip_init_length": Integer(0, None),
        "opt_skip_period": Integer(1, None),
        "opt_maxiter": Integer(1, None),
        "opt_nstarts": Integer(1, None),
        "opt_warmstart": Boolean(),
        "opt_verbose": Boolean(),
        "opt_debug_writer": Boolean(),
        "num_fantasy_samples": Integer(1, None),
        "num_init_random": Integer(0, None),
        "num_init_candidates": Integer(2, None),
        "initial_scoring": Categorical(choices=tuple(SUPPORTED_INITIAL_SCORING)),
        "skip_local_optimization": Boolean(),
        "debug_log": Boolean(),
        "normalize_targets": Boolean(),
        "no_fantasizing": Boolean(),
        "allow_duplicates": Boolean(),
        "input_warping": Boolean(),
        "boxcox_transform": Boolean(),
        "max_size_data_for_model": IntegerOrNone(1, None),
        "max_size_top_fraction": Float(0.0, 1.0),
        "gp_base_kernel": Categorical(choices=SUPPORTED_BASE_MODELS),
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


def gp_fifo_searcher_defaults(kwargs: Dict[str, Any]) -> (Set[str], dict, dict):
    """
    Returns ``mandatory``, ``default_options``, ``config_space`` for
    :func:`~syne_tune.optimizer.schedulers.searchers.utils.default_arguments.check_and_merge_defaults`
    to be applied to ``search_options`` for
    :class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher`.

    :return: ``(mandatory, default_options, config_space)``

    """
    return _common_defaults(
        kwargs,
        is_hyperband=False,
        is_restrict_configs=kwargs.get("restrict_configurations") is not None,
    )


def gp_multifidelity_searcher_defaults(
    kwargs: Dict[str, Any]
) -> (Set[str], dict, dict):
    """
    Returns ``mandatory``, ``default_options``, ``config_space`` for
    :func:`~syne_tune.optimizer.schedulers.searchers.utils.default_arguments.check_and_merge_defaults`
    to be applied to ``search_options`` for
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`.

    :return: ``(mandatory, default_options, config_space)``

    """
    return _common_defaults(
        kwargs,
        is_hyperband=True,
        is_restrict_configs=kwargs.get("restrict_configurations") is not None,
    )


def hypertune_searcher_defaults(kwargs: Dict[str, Any]) -> (Set[str], dict, dict):
    """
    Returns ``mandatory``, ``default_options``, ``config_space`` for
    :func:`~syne_tune.optimizer.schedulers.searchers.utils.default_arguments.check_and_merge_defaults`
    to be applied to ``search_options`` for
    :class:`~syne_tune.optimizer.schedulers.searchers.legacy_hypertune.HyperTuneSearcher`.

    :return: ``(mandatory, default_options, config_space)``

    """
    return _common_defaults(
        kwargs,
        is_hyperband=True,
        is_hypertune=True,
        is_restrict_configs=kwargs.get("restrict_configurations") is not None,
    )


def constrained_gp_fifo_searcher_defaults(
    kwargs: Dict[str, Any]
) -> (Set[str], dict, dict):
    """
    Returns ``mandatory``, ``default_options``, ``config_space`` for
    :func:`~syne_tune.optimizer.schedulers.searchers.utils.default_arguments.check_and_merge_defaults` to be applied to ``search_options`` for
    :class:`~syne_tune.optimizer.schedulers.searchers.legacy_constrained.ConstrainedGPFIFOSearcher`.

    :return: ``(mandatory, default_options, config_space)``

    """
    return _common_defaults(
        kwargs,
        is_hyperband=False,
        is_multi_output=True,
        is_restrict_configs=kwargs.get("restrict_configurations") is not None,
    )


def cost_aware_gp_fifo_searcher_defaults(
    kwargs: Dict[str, Any]
) -> (Set[str], dict, dict):
    """
    Returns ``mandatory``, ``default_options``, ``config_space`` for
    :func:`~syne_tune.optimizer.schedulers.searchers.utils.default_arguments.check_and_merge_defaults`
    to be applied to ``search_options`` for
    :class:`~syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.CostAwareGPFIFOSearcher`.

    :return: ``(mandatory, default_options, config_space)``

    """
    return _common_defaults(
        kwargs,
        is_hyperband=False,
        is_multi_output=True,
        is_restrict_configs=kwargs.get("restrict_configurations") is not None,
    )


def cost_aware_gp_multifidelity_searcher_defaults(
    kwargs: Dict[str, Any]
) -> (Set[str], dict, dict):
    """
    Returns ``mandatory``, ``default_options``, ``config_space`` for
    :func:`~syne_tune.optimizer.schedulers.searchers.utils.default_arguments.check_and_merge_defaults`
    to be applied to ``search_options`` for
    :class:`~syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.CostAwareGPMultiFidelitySearcher`.

    :return: ``(mandatory, default_options, config_space)``

    """
    return _common_defaults(
        kwargs,
        is_hyperband=True,
        is_multi_output=True,
        is_restrict_configs=kwargs.get("restrict_configurations") is not None,
    )
