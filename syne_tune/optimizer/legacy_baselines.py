from typing import Dict, Optional, Any, List, Union
import logging
import copy
from functools import partial

from syne_tune.optimizer.schedulers import (
    LegacyFIFOScheduler,
    LegacyHyperbandScheduler,
)
from syne_tune.optimizer.schedulers.legacy_pbt import LegacyPopulationBasedTraining
from syne_tune.optimizer.schedulers.multiobjective import (
    LegacyMOASHA,
    NSGA2Searcher,
    LegacyLinearScalarizedScheduler,
)
from syne_tune.optimizer.schedulers.multiobjective.legacy_multi_objective_regularized_evolution import (
    LegacyMultiObjectiveRegularizedEvolution,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator import Estimator
from syne_tune.optimizer.schedulers.searchers.legacy_regularized_evolution import (
    LegacyRegularizedEvolution,
)
from syne_tune.optimizer.schedulers.synchronous import (
    SynchronousGeometricHyperbandScheduler,
    GeometricDifferentialEvolutionHyperbandScheduler,
)
from syne_tune.optimizer.schedulers.transfer_learning import (
    LegacyTransferLearningTaskEvaluations,
)
from syne_tune.optimizer.schedulers.random_seeds import RandomSeedGenerator

from syne_tune.util import dict_get

logger = logging.getLogger(__name__)


def _random_seed_from_generator(random_seed: int) -> int:
    """
    This helper makes sure that a searcher within
    :class:`~syne_tune.optimizer.schedulers.FIFOScheduler` is seeded in the same
    way whether it is created by the searcher factory, or by hand.

    :param random_seed: Random seed for scheduler
    :return: Random seed to be used for searcher created by hand
    """
    return RandomSeedGenerator(random_seed)()


def _assert_searcher_must_be(kwargs: Dict[str, Any], name: str):
    searcher = kwargs.get("searcher")
    assert searcher is None or searcher == name, f"Must have searcher='{name}'"


class RandomSearch(LegacyFIFOScheduler):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(self, config_space: Dict[str, Any], metric: str, **kwargs):
        searcher_name = "random"
        _assert_searcher_must_be(kwargs, searcher_name)
        super(RandomSearch, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            **kwargs,
        )


class GridSearch(LegacyFIFOScheduler):
    """Grid search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.GridSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(self, config_space: Dict[str, Any], metric: str, **kwargs):
        searcher_name = "grid"
        _assert_searcher_must_be(kwargs, searcher_name)
        super(GridSearch, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            **kwargs,
        )


class BayesianOptimization(LegacyFIFOScheduler):
    """Gaussian process based Bayesian optimization.

    See :class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(self, config_space: Dict[str, Any], metric: str, **kwargs):
        searcher_name = "bayesopt"
        _assert_searcher_must_be(kwargs, searcher_name)
        super(BayesianOptimization, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            **kwargs,
        )


def _assert_need_one(kwargs: Dict[str, Any], need_one: Optional[set] = None):
    if need_one is None:
        need_one = {"max_t", "max_resource_attr"}
    assert need_one.intersection(kwargs.keys()), f"Need one of these: {need_one}"


class ASHA(LegacyHyperbandScheduler):
    """Asynchronous Sucessive Halving (ASHA).

    One of ``max_t``, ``max_resource_attr`` needs to be in ``kwargs``. For
    ``type="promotion"``, the latter is more useful.

    See also :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` for
    ``kwargs`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param kwargs: Additional arguments to :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    """

    def __init__(
        self, config_space: Dict[str, Any], metric: str, resource_attr: str, **kwargs
    ):
        _assert_need_one(kwargs)
        searcher_name = "random"
        _assert_searcher_must_be(kwargs, searcher_name)
        super(ASHA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            resource_attr=resource_attr,
            **kwargs,
        )


class MOBSTER(LegacyHyperbandScheduler):
    """Model-based Asynchronous Multi-fidelity Optimizer (MOBSTER).

    One of ``max_t``, ``max_resource_attr`` needs to be in ``kwargs``. For
    ``type="promotion"``, the latter is more useful, see also
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    MOBSTER can be run with different surrogate models. The model is selected
    by ``search_options["model"]`` in ``kwargs``. The default is ``"gp_multitask"``
    (jointly dependent multi-task GP model), another useful choice is
    ``"gp_independent"`` (independent GP models at each rung level, with shared
    ARD kernel).

    See also:

    * :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` for ``kwargs``
      parameters
    * :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`
      for ``kwargs["search_options"]`` parameters

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param kwargs: Additional arguments to :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    """

    def __init__(
        self, config_space: Dict[str, Any], metric: str, resource_attr: str, **kwargs
    ):
        _assert_need_one(kwargs)
        searcher_name = "bayesopt"
        _assert_searcher_must_be(kwargs, searcher_name)
        super(MOBSTER, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            resource_attr=resource_attr,
            **kwargs,
        )


class HyperTune(LegacyHyperbandScheduler):
    """
    One of ``max_t``, ``max_resource_attr`` needs to be in ``kwargs``. For
    ``type="promotion"``, the latter is more useful, see also
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    Hyper-Tune is a model-based variant of ASHA with more than one bracket.
    It can be seen as extension of MOBSTER and can be used with
    ``search_options["model"]`` in ``kwargs`` being ``"gp_independent"`` or
    ``"gp_multitask"``. It has a model-based way to sample the bracket for every
    new trial, as well as an ensemble predictive distribution feeding into the
    acquisition function. Our implementation is based on:

        | Yang Li et al
        | Hyper-Tune: Towards Efficient Hyper-parameter Tuning at Scale
        | VLDB 2022
        | https://arxiv.org/abs/2201.06834

    See also:

    * :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` for ``kwargs``
      parameters
    * :class:`~syne_tune.optimizer.schedulers.searchers.legacy_hypertune.HyperTuneSearcher`
      for ``kwargs["search_options"]`` parameters
    * :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.legacy_hypertune.gp_model.HyperTuneIndependentGPModel`
      for implementation

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param kwargs: Additional arguments to :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    """

    def __init__(self, config_space: Dict, metric: str, resource_attr: str, **kwargs):
        _assert_need_one(kwargs)
        searcher_name = "legacy_hypertune"
        _assert_searcher_must_be(kwargs, searcher_name)
        kwargs = copy.deepcopy(kwargs)
        search_options = dict_get(kwargs, "search_options", dict())
        k, v, supp = "model", "gp_independent", {"gp_independent", "gp_multitask"}
        model = search_options.get(k, v)
        assert model in supp, (
            f"HyperTune does not support search_options['{k}'] = '{model}'"
            f", must be in {supp}"
        )
        search_options[k] = model
        kwargs["search_options"] = search_options
        super(HyperTune, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            resource_attr=resource_attr,
            **kwargs,
        )


class DyHPO(LegacyHyperbandScheduler):
    """Dynamic Gray-Box Hyperparameter Optimization (DyHPO)

    One of ``max_t``, ``max_resource_attr`` needs to be in ``kwargs``. The latter
    is more useful (DyHPO is a pause-resume scheduler), see also
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    DyHPO can be run with the same surrogate models as :class:`MOBSTER`, but
    ``search_options["model"] != "gp_independent"``. This is because DyHPO
    requires extrapolation to resource levels without any data, which cannot
    sensibly be done with independent GPs per resource level. Compared to
    :class:`MOBSTER` or :class:`HyperTune`, DyHPO is typically run with linearly
    spaced rung levels (the default being 1, 2, 3, ...). Decisions whether to
    promote a paused trial are folded together with suggesting a new
    configuration, both are model-based. Our implementation is based on

        | Wistuba, M. and Kadra, A. and Grabocka, J.
        | Dynamic and Efficient Gray-Box Hyperparameter Optimization for Deep Learning
        | https://arxiv.org/abs/2202.09774

    However, there are important differences:

    * We do not implement their surrogate model based on a neural network kernel,
      but instead just use the surrogate models we provide for :class:`MOBSTER` as
      well
    * We implement a hybrid of DyHPO with the asynchronous successive halving
      rule for promoting trials, controlled by ``probability_sh``. With this
      probability, we promote a trial via the SH rule. This mitigates the issue
      that DyHPO tends to start many trials initially, because due to lack of any
      data at higher rungs, the score values for promoting a trial are much worse
      than those for starting a new one.

    See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` for ``kwargs``
    parameters, and
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`
    for ``kwargs["search_options"]`` parameters. The following parameters are
    most important for DyHPO:

    * ``rung_increment`` (and ``grace_period``): These parameters determine the
      rung level spacing. DyHPO is run with linearly spaced rung levels
      :math:`r_{min} + k \nu`, where :math:`r_{min}` is ``grace_period`` and
      :math:`\nu` is ``rung_increment``. The default is 2.
    * ``probability_sh``: See comment. The smaller this probability, the closer
      the method is to the published original, which tends to start many more
      trials than promote paused ones. On the other hand, if this probability is
      close to 1, you may as well run MOBSTER. The default is
      :const:`~syne_tune.optimizer.schedulers.searchers.legacy_dyhpo.hyperband_dyhpo.DEFAULT_SH_PROBABILITY`.
    * ``search_options["opt_skip_period"]``: DyHPO can be quite a bit slower
      than MOBSTER, because the GP surrogate model is used more frequently. It
      can be sped up a bit by changing ``opt_skip_period`` (general default is
      1). The default here is 3.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param probability_sh: See above
    :param kwargs: Additional arguments to :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        resource_attr: str,
        probability_sh: Optional[float] = None,
        **kwargs,
    ):
        _assert_need_one(kwargs)
        searcher_name = "legacy_dyhpo"
        _assert_searcher_must_be(kwargs, searcher_name)
        scheduler_type = kwargs.get("type")
        assert (
            scheduler_type is None or scheduler_type == "legacy_dyhpo"
        ), "Must have type='legacy_dyhpo'"
        kwargs["type"] = "legacy_dyhpo"
        if probability_sh is not None:
            rung_system_kwargs = dict_get(kwargs, "rung_system_kwargs", dict())
            rung_system_kwargs["probability_sh"] = probability_sh
            kwargs["rung_system_kwargs"] = rung_system_kwargs
        search_options = dict_get(kwargs, "search_options", dict())
        k = "opt_skip_period"
        if k not in search_options:
            search_options[k] = 3
        kwargs["search_options"] = search_options
        if (
            kwargs.get("reduction_factor") is None
            and kwargs.get("rung_increment") is None
        ):
            kwargs["rung_increment"] = 2
        super(DyHPO, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            resource_attr=resource_attr,
            **kwargs,
        )


class PASHA(LegacyHyperbandScheduler):
    """Progressive ASHA.

    One of ``max_t``, ``max_resource_attr`` needs to be in ``kwargs``. The latter is
    more useful, see also :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param kwargs: Additional arguments to :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    """

    def __init__(
        self, config_space: Dict[str, Any], metric: str, resource_attr: str, **kwargs
    ):
        _assert_need_one(kwargs)
        super(PASHA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",  # default, can be overwritten
            resource_attr=resource_attr,
            type="pasha",
            **kwargs,
        )


class BOHB(LegacyHyperbandScheduler):
    """Asynchronous BOHB

    Combines :class:`ASHA` with TPE-like Bayesian optimization, using kernel
    density estimators.

    One of ``max_t``, ``max_resource_attr`` needs to be in ``kwargs``. For
    ``type="promotion"``, the latter is more useful, see also
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    See
    :class:`~syne_tune.optimizer.schedulers.searchers.kde.MultiFidelityKernelDensityEstimator`
    for ``kwargs["search_options"]`` parameters, and
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` for ``kwargs``
    parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param kwargs: Additional arguments to :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    """

    def __init__(
        self, config_space: Dict[str, Any], metric: str, resource_attr: str, **kwargs
    ):
        _assert_need_one(kwargs)
        searcher_name = "kde"
        _assert_searcher_must_be(kwargs, searcher_name)
        super(BOHB, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            resource_attr=resource_attr,
            **kwargs,
        )


class SyncHyperband(SynchronousGeometricHyperbandScheduler):
    """Synchronous Hyperband.

    One of ``max_resource_level``, ``max_resource_attr`` needs to be in ``kwargs``.
    The latter is more useful, see also :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    If ``kwargs["brackets"]`` is not given, the maximum number of brackets is
    used. Choose ``kwargs["brackets"] = 1`` for synchronous successive halving.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousGeometricHyperbandScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        resource_attr: str,
        **kwargs,
    ):
        _assert_need_one(kwargs, need_one={"max_resource_level", "max_resource_attr"})
        super(SyncHyperband, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",  # default, can be overwritten
            resource_attr=resource_attr,
            **kwargs,
        )


class SyncBOHB(SynchronousGeometricHyperbandScheduler):
    """Synchronous BOHB.

    Combines :class:`SyncHyperband` with TPE-like Bayesian optimization, using
    kernel density estimators.

    One of ``max_resource_level``, ``max_resource_attr`` needs to be in ``kwargs``.
    The latter is more useful, see also
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    If ``kwargs["brackets"]`` is not given, the maximum number of brackets is
    used. Choose ``kwargs["brackets"] = 1`` for synchronous successive halving.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousGeometricHyperbandScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        resource_attr: str,
        **kwargs,
    ):
        _assert_need_one(kwargs, need_one={"max_resource_level", "max_resource_attr"})
        searcher_name = "kde"
        _assert_searcher_must_be(kwargs, searcher_name)
        super(SyncBOHB, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            resource_attr=resource_attr,
            **kwargs,
        )


class DEHB(GeometricDifferentialEvolutionHyperbandScheduler):
    """Differential Evolution Hyperband (DEHB).

    Combines :class:`SyncHyperband` with ideas from evolutionary algorithms.

    One of ``max_resource_level``, ``max_resource_attr`` needs to be in ``kwargs``.
    The latter is more useful, see also :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousGeometricHyperbandScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        resource_attr: str,
        **kwargs,
    ):
        _assert_need_one(kwargs, need_one={"max_resource_level", "max_resource_attr"})
        super(DEHB, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random_encoded",  # default, can be overwritten
            resource_attr=resource_attr,
            **kwargs,
        )


class SyncMOBSTER(SynchronousGeometricHyperbandScheduler):
    """Synchronous MOBSTER.

    Combines :class:`SyncHyperband` with Gaussian process based Bayesian
    optimization, just like :class:`MOBSTER` builds on top of :class:`ASHA` in
    the asynchronous case.

    One of ``max_resource_level``, ``max_resource_attr`` needs to be in ``kwargs``.
    The latter is more useful, see also
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.

    If ``kwargs["brackets"]`` is not given, the maximum number of brackets is
    used. Choose ``kwargs["brackets"] = 1`` for synchronous successive halving.

    The default surrogate model (``search_options["model"]`` in ``kwargs``) is
    ``"gp_independent"``, different to :class:`MOBSTER`.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousGeometricHyperbandScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        resource_attr: str,
        **kwargs,
    ):
        _assert_need_one(kwargs, need_one={"max_resource_level", "max_resource_attr"})
        searcher_name = "bayesopt"
        _assert_searcher_must_be(kwargs, searcher_name)
        search_options = dict_get(kwargs, "search_options", dict())
        if "model" not in search_options:
            search_options["model"] = "gp_independent"
        kwargs["search_options"] = search_options
        super(SyncMOBSTER, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            resource_attr=resource_attr,
            **kwargs,
        )


def _create_searcher_kwargs(
    config_space: Dict[str, Any],
    metric: Union[str, List[str]],
    random_seed: Optional[int],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    searcher_kwargs = dict(
        config_space=config_space,
        metric=metric,
        points_to_evaluate=kwargs.get("points_to_evaluate"),
    )
    search_options = dict_get(kwargs, "search_options", dict())
    searcher_kwargs.update(search_options)
    if random_seed is not None:
        searcher_kwargs["random_seed"] = _random_seed_from_generator(random_seed)
    return searcher_kwargs


class BORE(LegacyFIFOScheduler):
    """Bayesian Optimization by Density-Ratio Estimation (BORE).

    See :class:`~syne_tune.optimizer.schedulers.searchers.bore.Bore`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param random_seed: Random seed, optional
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        try:
            from syne_tune.optimizer.schedulers.searchers.bore.legacy_bore import (
                LegacyBore,
            )
        except ImportError:
            raise

        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        super(BORE, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=LegacyBore(**searcher_kwargs),
            random_seed=random_seed,
            **kwargs,
        )


class ASHABORE(LegacyHyperbandScheduler):
    """Model-based ASHA with BORE searcher

    See :class:`~syne_tune.optimizer.schedulers.searchers.bore.MultiFidelityBore`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param random_seed: Random seed, optional
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        resource_attr: str,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        try:
            from syne_tune.optimizer.schedulers.searchers.bore.legacy_multi_fidelity_bore import (
                LegacyMultiFidelityBore,
            )
        except ImportError:
            raise

        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        searcher_kwargs["resource_attr"] = resource_attr
        searcher_kwargs["mode"] = kwargs.get("mode")
        super(ASHABORE, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=LegacyMultiFidelityBore(**searcher_kwargs),
            resource_attr=resource_attr,
            random_seed=random_seed,
            **kwargs,
        )


class BoTorch(LegacyFIFOScheduler):
    """Bayesian Optimization using BoTorch

    See :class:`~syne_tune.optimizer.schedulers.searchers.botorch.BoTorchSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param random_seed: Random seed, optional
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        try:
            from syne_tune.optimizer.schedulers.searchers.botorch.legacy_botorch_searcher import (
                LegacyBoTorchSearcher,
            )
        except ImportError:
            raise

        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        super(BoTorch, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=LegacyBoTorchSearcher(**searcher_kwargs),
            random_seed=random_seed,
            **kwargs,
        )


class REA(LegacyFIFOScheduler):
    """Regularized Evolution (REA).

    See :class:`~syne_tune.optimizer.schedulers.searchers.regularized_evolution.RegularizedEvolution`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param population_size: See
        :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
        Defaults to 100
    :param sample_size: See
        :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
        Defaults to 10
    :param random_seed: Random seed, optional
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        population_size: int = 100,
        sample_size: int = 10,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        searcher_kwargs["population_size"] = population_size
        searcher_kwargs["sample_size"] = sample_size
        super(REA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=LegacyRegularizedEvolution(**searcher_kwargs),
            random_seed=random_seed,
            **kwargs,
        )


def create_gaussian_process_estimator(
    config_space: Dict[str, Any],
    metric: str,
    random_seed: Optional[int] = None,
    search_options: Optional[Dict[str, Any]] = None,
) -> Estimator:
    scheduler = BayesianOptimization(
        config_space=config_space,
        metric=metric,
        random_seed=random_seed,
        search_options=search_options,
    )
    searcher = scheduler.searcher  # GPFIFOSearcher
    state_transformer = searcher.state_transformer  # ModelStateTransformer
    estimator = state_transformer.estimator  # GaussProcEmpiricalBayesEstimator

    # update the estimator properties
    estimator.active_metric = metric
    return estimator


class MORandomScalarizationBayesOpt(LegacyFIFOScheduler):
    """
    Uses :class:`~syne_tune.optimizer.schedulers.multiobjective.MultiObjectiveMultiSurrogateSearcher`
    with one standard GP surrogate model per metric (same as in
    :class:`BayesianOptimization`, together with the
    :class:`~syne_tune.optimizer.schedulers.multiobjective.MultiObjectiveLCBRandomLinearScalarization`
    acquisition function.

    If `estimators` is given, surrogate models are taken from there, and the
    default is used otherwise. This is useful if you have a good low-variance
    model for one of the objectives.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metrics to optimize
    :param mode: Modes of optimization. Defaults to "min" for all
    :param random_seed: Random seed, optional
    :param estimators: Use these surrogate models instead of the default GP
        one. Optional
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. Here,
        ``kwargs["search_options"]`` is used to create the searcher and its
        GP surrogate models.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: List[str],
        mode: Union[List[str], str] = "min",
        random_seed: Optional[int] = None,
        estimators: Optional[Dict[str, Estimator]] = None,
        **kwargs,
    ):
        try:
            from syne_tune.optimizer.schedulers.multiobjective import (
                MultiObjectiveMultiSurrogateSearcher,
                MultiObjectiveLCBRandomLinearScalarization,
            )
        except ImportError:
            raise

        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )

        if estimators is None:
            estimators = dict()
        else:
            estimators = estimators.copy()
        if isinstance(mode, str):
            mode = [mode] * len(metric)
        if "search_options" in kwargs:
            search_options = kwargs["search_options"].copy()
        else:
            search_options = dict()
        search_options["no_fantasizing"] = True
        for _metric in metric:
            if _metric not in estimators:
                estimators[_metric] = create_gaussian_process_estimator(
                    config_space=config_space,
                    metric=_metric,
                    search_options=search_options,
                )
        # Note: ``mode`` is dealt with in the ``update`` method of the MO
        # searcher, by converting the metrics. Internally, all metrics are
        # minimized
        searcher = MultiObjectiveMultiSurrogateSearcher(
            estimators=estimators,
            mode=mode,
            scoring_class=partial(
                MultiObjectiveLCBRandomLinearScalarization, random_seed=random_seed
            ),
            **searcher_kwargs,
        )
        super().__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=searcher,
            random_seed=random_seed,
            **kwargs,
        )


class NSGA2(LegacyFIFOScheduler):
    """
    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param population_size: The size of the population for NSGA-2
    :param random_seed: Random seed, optional
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: List[str],
        mode: Union[List[str], str] = "min",
        population_size: int = 20,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        searcher_kwargs["mode"] = mode
        searcher_kwargs["population_size"] = population_size
        super(NSGA2, self).__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=NSGA2Searcher(**searcher_kwargs),
            random_seed=random_seed,
            **kwargs,
        )


class MOREA(LegacyFIFOScheduler):
    """

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param population_size: See
        :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
        Defaults to 100
    :param sample_size: See
        :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
        Defaults to 10
    :param random_seed: Random seed, optional
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: List[str],
        mode: Union[List[str], str] = "min",
        population_size: int = 100,
        sample_size: int = 10,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        searcher_kwargs["mode"] = mode
        searcher_kwargs["population_size"] = population_size
        searcher_kwargs["sample_size"] = sample_size
        super(MOREA, self).__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=LegacyMultiObjectiveRegularizedEvolution(**searcher_kwargs),
            random_seed=random_seed,
            **kwargs,
        )


class MOLinearScalarizationBayesOpt(LegacyLinearScalarizedScheduler):
    """
    Uses :class:`~syne_tune.optimizer.schedulers.multiobjective.LinearScalarizedScheduler`
    together with a default GP surrogate model.

    See :class:`~syne_tune.optimizer.schedulers.searchers.GPFIFOSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param scalarization_weights: Positive weight used for the scalarization.
        Defaults to all 1
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: List[str],
        scalarization_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        searcher_name = "bayesopt"
        _assert_searcher_must_be(kwargs, searcher_name)
        super().__init__(
            config_space=config_space,
            metric=metric,
            scalarization_weights=scalarization_weights,
            searcher=searcher_name,
            **kwargs,
        )


class ConstrainedBayesianOptimization(LegacyFIFOScheduler):
    """Constrained Bayesian Optimization.

    See :class:`~syne_tune.optimizer.schedulers.searchers.legacy_constrained.ConstrainedGPFIFOSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param constraint_attr: Name of constraint metric
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self, config_space: Dict[str, Any], metric: str, constraint_attr: str, **kwargs
    ):
        searcher_name = "bayesopt_constrained"
        _assert_searcher_must_be(kwargs, searcher_name)
        search_options = dict_get(kwargs, "search_options", dict())
        kwargs["search_options"] = dict(search_options, constraint_attr=constraint_attr)
        super(ConstrainedBayesianOptimization, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            **kwargs,
        )


class ZeroShotTransfer(LegacyFIFOScheduler):
    """
    A zero-shot transfer hyperparameter optimization method which jointly selects configurations that minimize the
    average rank obtained on historic metadata (transfer_learning_evaluations).
    Reference:

        | Sequential Model-Free Hyperparameter Tuning.
        | Martin Wistuba, Nicolas Schilling, Lars Schmidt-Thieme.
        | IEEE International Conference on Data Mining (ICDM) 2015.

    :param config_space: Configuration space for evaluation function
    :param transfer_learning_evaluations: Dictionary from task name to offline
        evaluations.
    :param metric: Name of metric to optimize
    :param mode: Whether to minimize (min) or maximize (max)
    :param sort_transfer_learning_evaluations: Use ``False`` if the
        hyperparameters for each task in ``transfer_learning_evaluations`` are
        already in the same order. If set to ``True``, hyperparameters are sorted.
    :param use_surrogates: If the same configuration is not evaluated on all
        tasks, set this to ``True``. This will generate a set of configurations
        and will impute their performance using surrogate models.
    :param random_seed: Used for randomly sampling candidates. Only used if
        ``use_surrogates=True``.
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        transfer_learning_evaluations: Dict[str, LegacyTransferLearningTaskEvaluations],
        metric: str,
        mode: str = "min",
        sort_transfer_learning_evaluations: bool = True,
        use_surrogates: bool = False,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        try:
            from syne_tune.optimizer.schedulers.transfer_learning import (
                legacy_zero_shot,
            )
        except ImportError:
            raise

        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        searcher_kwargs.update(
            dict(
                mode=mode,
                sort_transfer_learning_evaluations=sort_transfer_learning_evaluations,
                transfer_learning_evaluations=transfer_learning_evaluations,
                use_surrogates=use_surrogates,
            )
        )
        super(ZeroShotTransfer, self).__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=legacy_zero_shot.LegacyZeroShotTransfer(**searcher_kwargs),
            random_seed=random_seed,
            **kwargs,
        )


class ASHACTS(LegacyHyperbandScheduler):
    """
    Runs ASHA where the searcher is done with the transfer-learning method:

        | A Quantile-based Approach for Hyperparameter Transfer Learning.
        | David Salinas, Huibin Shen, Valerio Perrone.
        | ICML 2020.

    This is the Copula Thompson Sampling approach described in the paper where
    a surrogate is fitted on the transfer learning data to predict mean and
    variance of configuration performance given a hyperparameter. The surrogate
    is then sampled from, and the best configurations are returned as next
    candidate to evaluate.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param resource_attr: Name of resource attribute
    :param transfer_learning_evaluations: Dictionary from task name to offline
        evaluations.
    :param mode: Whether to minimize (min) or maximize (max)
    :param random_seed: Used for randomly sampling candidates
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        resource_attr: str,
        transfer_learning_evaluations: Dict[str, LegacyTransferLearningTaskEvaluations],
        mode: str = "min",
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        try:
            from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.legacy_quantile_based_searcher import (
                LegacyQuantileBasedSurrogateSearcher,
            )
        except ImportError:
            raise

        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        searcher_kwargs.update(
            dict(
                mode=mode,
                transfer_learning_evaluations=transfer_learning_evaluations,
            )
        )
        super(ASHACTS, self).__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=LegacyQuantileBasedSurrogateSearcher(**searcher_kwargs),
            resource_attr=resource_attr,
            random_seed=random_seed,
            **kwargs,
        )


class KDE(LegacyFIFOScheduler):
    """Single-fidelity variant of BOHB

    Combines :class:`~syne_tune.optimizer.schedulers.FIFOScheduler` with TPE-like
    Bayesian optimization, using kernel density estimators.

    See
    :class:`~syne_tune.optimizer.schedulers.searchers.kde.KernelDensityEstimator`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(self, config_space: Dict[str, Any], metric: str, **kwargs):
        searcher_name = "kde"
        _assert_searcher_must_be(kwargs, searcher_name)
        super(KDE, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            **kwargs,
        )


class CQR(LegacyFIFOScheduler):
    """
    Single-fidelity Conformal Quantile Regression approach proposed in:
        | Optimizing Hyperparameters with Conformal Quantile Regression.
        | David Salinas, Jacek Golebiowski, Aaron Klein, Matthias Seeger, Cedric Archambeau.
        | ICML 2023.
    The method predict quantile performance with gradient boosted trees and calibrate prediction with conformal
    predictions.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        mode: str = "min",
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        try:
            from syne_tune.optimizer.schedulers.searchers.conformal.legacy_surrogate_searcher import (
                LegacySurrogateSearcher,
            )
        except ImportError:
            raise

        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        searcher_kwargs["mode"] = mode
        super(CQR, self).__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=LegacySurrogateSearcher(**searcher_kwargs),
            random_seed=random_seed,
            **kwargs,
        )


class ASHACQR(LegacyHyperbandScheduler):
    """
    Multi-fidelity Conformal Quantile Regression approach proposed in:
        | Optimizing Hyperparameters with Conformal Quantile Regression.
        | David Salinas, Jacek Golebiowski, Aaron Klein, Matthias Seeger, Cedric Archambeau.
        | ICML 2023.
    The method predict quantile performance with gradient boosted trees and calibrate prediction with conformal
    predictions.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        resource_attr: str,
        mode: str = "min",
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        try:
            from syne_tune.optimizer.schedulers.searchers.conformal.legacy_surrogate_searcher import (
                LegacySurrogateSearcher,
            )
        except ImportError:
            raise

        searcher_kwargs = _create_searcher_kwargs(
            config_space, metric, random_seed, kwargs
        )
        searcher_kwargs["mode"] = mode
        super(ASHACQR, self).__init__(
            config_space=config_space,
            metric=metric,
            mode=mode,
            searcher=LegacySurrogateSearcher(**searcher_kwargs),
            resource_attr=resource_attr,
            random_seed=random_seed,
            **kwargs,
        )


try:
    from syne_tune.optimizer.schedulers.legacy_smac_scheduler import LegacySMACScheduler

    class SMAC(LegacySMACScheduler):
        """
        Wrapper to SMAC3. Requires SMAC3 to be installed, see https://github.com/automl/SMAC3 for instructions.
        """

        def __init__(
            self,
            config_space: Dict[str, Any],
            metric: str,
            mode: str = "min",
            points_to_evaluate: Optional[list] = None,
            random_seed: Optional[int] = None,
            **kwargs,
        ):
            """
            :param config_space:
            :param metric: metric to be optimized, should be present in reported results dictionary
            :param mode: "min" or "max", default to "min"
            :param points_to_evaluate: list of points to consider before calling the optimizer
            :param random_seed: to fix the behavior of smac
            """
            super(SMAC, self).__init__(
                config_space=config_space,
                metric=metric,
                mode=mode,
                random_seed=random_seed,
                points_to_evaluate=points_to_evaluate,
                **kwargs,
            )

except ImportError as e:
    logging.debug(e)

try:
    from syne_tune.optimizer.schedulers.multiobjective.legacy_expected_hyper_volume_improvement import (
        LegacyExpectedHyperVolumeImprovement,
    )

    class EHVI(LegacyFIFOScheduler):
        """
        Implements the Expected Hypervolume Improvement method.

        See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
        for ``kwargs["search_options"]`` parameters.

        :param config_space: Configuration space for evaluation function
        :param metric: Name of metric to optimize
        :param population_size: See
            :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
            Defaults to 100
        :param sample_size: See
            :class:`~syne_tune.optimizer.schedulers.searchers.RegularizedEvolution`.
            Defaults to 10
        :param random_seed: Random seed, optional
        :param kwargs: Additional arguments to
            :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
        """

        def __init__(
            self,
            config_space: Dict[str, Any],
            metric: List[str],
            mode: Union[List[str], str] = "min",
            random_seed: Optional[int] = None,
            **kwargs,
        ):
            searcher_kwargs = _create_searcher_kwargs(
                config_space, metric, random_seed, kwargs
            )
            searcher_kwargs["mode"] = mode
            print(mode)

            super(EHVI, self).__init__(
                config_space=config_space,
                metric=metric,
                mode=mode,
                searcher=LegacyExpectedHyperVolumeImprovement(**searcher_kwargs),
                random_seed=random_seed,
                **kwargs,
            )

except ImportError as e:
    logging.debug(e)
# Dictionary that allows to also list baselines who don't need a wrapper class
# such as :class:`PopulationBasedTraining`
baselines_dict = {
    "Random Search": RandomSearch,
    "Grid Search": GridSearch,
    "Bayesian Optimization": BayesianOptimization,
    "ASHA": ASHA,
    "MOBSTER": MOBSTER,
    "PASHA": PASHA,
    "MOASHA": LegacyMOASHA,
    "PBT": LegacyPopulationBasedTraining,
    "BORE": BORE,
    "REA": REA,
    "SyncHyperband": SyncHyperband,
    "SyncBOHB": SyncBOHB,
    "DEHB": DEHB,
    "SyncMOBSTER": SyncMOBSTER,
    "ConstrainedBayesianOptimization": ConstrainedBayesianOptimization,
    "ZeroShotTransfer": ZeroShotTransfer,
    "ASHACTS": ASHACTS,
    "CQR": CQR,
    "ASHACQR": ASHACQR,
}
