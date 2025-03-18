import logging

from syne_tune.optimizer.schedulers.searchers import (
    LegacyRandomSearcher,
    GridSearcher,
)
from syne_tune.optimizer.schedulers.searchers.legacy_searcher import LegacyBaseSearcher

__all__ = ["legacy_searcher_factory"]

logger = logging.getLogger(__name__)


SUPPORTED_SEARCHERS_FIFO = {
    "random",
    "grid",
    "kde",
    "cqr",
    "botorch",
    "bayesopt",
    "bayesopt_constrained",
    "bayesopt_cost",
}


SUPPORTED_SEARCHERS_HYPERBAND = {
    "random",
    "grid",
    "kde",
    "cqr",
    "bayesopt",
    "bayesopt_cost",
    "legacy_hypertune",
    "legacy_dyhpo",
}


_OUR_MULTIFIDELITY_SCHEDULERS = {
    "hyperband_stopping",
    "hyperband_promotion",
    "hyperband_cost_promotion",
    "hyperband_pasha",
    "hyperband_synchronous",
}


def legacy_searcher_factory(searcher_name: str, **kwargs) -> LegacyBaseSearcher:
    """Factory for searcher objects

    This function creates searcher objects from string argument name and
    additional kwargs. It is typically called in the constructor of a
    scheduler (see :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`),
    which provides most of the required ``kwargs``.

    :param searcher_name: Value of ``searcher`` argument to scheduler (see
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`)
    :param kwargs: Argument to
        :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher` constructor
    :return: New searcher object
    """
    supported_schedulers = None
    scheduler = kwargs.get("scheduler")
    model = kwargs.get("model", "gp_multitask")
    if searcher_name == "random":
        searcher_cls = LegacyRandomSearcher
    elif searcher_name == "grid":
        searcher_cls = GridSearcher
    elif searcher_name == "kde":
        try:
            from syne_tune.optimizer.schedulers.searchers.kde.legacy_kde_searcher import (
                LegacyKernelDensityEstimator,
            )
            from syne_tune.optimizer.schedulers.searchers.kde.legacy_multi_fidelity_searcher import (
                LegacyMultiFidelityKernelDensityEstimator,
            )
        except ImportError:
            raise

        if scheduler == "fifo":
            searcher_cls = LegacyKernelDensityEstimator
        else:
            supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
            searcher_cls = LegacyMultiFidelityKernelDensityEstimator
    elif searcher_name == "cqr":
        try:
            from syne_tune.optimizer.schedulers.searchers.conformal.legacy_surrogate_searcher import (
                LegacySurrogateSearcher,
            )
        except ImportError:
            raise
        searcher_cls = LegacySurrogateSearcher
    elif searcher_name == "botorch":
        try:
            from syne_tune.optimizer.schedulers.searchers.botorch import (
                BoTorchSearcher,
            )
        except ImportError:
            raise

        searcher_cls = BoTorchSearcher
        supported_schedulers = {"fifo"}
    else:
        gp_searchers = {
            "bayesopt",
            "bayesopt_constrained",
            "bayesopt_cost",
            "legacy_hypertune",
            "legacy_dyhpo",
        }
        assert (
            searcher_name in gp_searchers
        ), f"searcher '{searcher_name}' is not supported"
        try:
            from syne_tune.optimizer.schedulers.searchers import (
                GPFIFOSearcher,
                GPMultiFidelitySearcher,
            )
            from syne_tune.optimizer.schedulers.searchers.legacy_constrained import (
                ConstrainedGPFIFOSearcher,
            )
            from syne_tune.optimizer.schedulers.searchers.legacy_cost_aware import (
                CostAwareGPFIFOSearcher,
                CostAwareGPMultiFidelitySearcher,
            )
            from syne_tune.optimizer.schedulers.searchers.legacy_hypertune import (
                HyperTuneSearcher,
            )
            from syne_tune.optimizer.schedulers.searchers.legacy_dyhpo import (
                DynamicHPOSearcher,
            )
        except ImportError:
            raise

        if searcher_name == "bayesopt":
            if scheduler == "fifo":
                searcher_cls = GPFIFOSearcher
            else:
                supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
                if (
                    model == "gp_multitask"
                    and kwargs.get("gp_resource_kernel") == "freeze-thaw"
                ):
                    logger.warning(
                        "You are combining model = gp_multitask with "
                        "gp_resource_kernel = freeze-thaw. This is mainly "
                        "for debug purposes. The same surrogate model is "
                        "obtained with model = gp_expdecay, but computations "
                        "are faster then."
                    )
                searcher_cls = GPMultiFidelitySearcher
        elif searcher_name == "legacy_hypertune":
            supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
            searcher_cls = HyperTuneSearcher
        elif searcher_name == "bayesopt_constrained":
            supported_schedulers = {"fifo"}
            searcher_cls = ConstrainedGPFIFOSearcher
        elif searcher_name == "legacy_dyhpo":
            supported_schedulers = {"hyperband_dyhpo", "hyperband_legacy_dyhpo"}
            searcher_cls = DynamicHPOSearcher
        else:  # bayesopt_cost
            if scheduler == "fifo":
                searcher_cls = CostAwareGPFIFOSearcher
            else:
                supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
                searcher_cls = CostAwareGPMultiFidelitySearcher

    if supported_schedulers is not None:
        assert scheduler is not None, "Scheduler must set search_options['scheduler']"
        assert scheduler in supported_schedulers, (
            f"Searcher '{searcher_name}' only works with schedulers "
            + f"{supported_schedulers} (not with '{scheduler}')"
        )
    searcher = searcher_cls(**kwargs)
    return searcher
