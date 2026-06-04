from typing import Any

from syne_tune.optimizer.schedulers.searchers.random_searcher import RandomSearcher
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher

# TODO better way to support listing available searchers and tie strings
searchers = [
    "random_search",
    "bore",
    "kde",
    "regularized_evolution",
    "cqr",
    "botorch",
]


def searcher_cls(searcher_name: str, is_multi_objective: bool = False):
    """
    :param searcher_name:
    :param is_multi_objective: Whether the searcher needs to support multi-objective optimization
    :return: the class associated to the searcher string
    """
    if searcher_name == "random_search":
        if is_multi_objective:
            from syne_tune.optimizer.schedulers.searchers.random_searcher import (
                MultiObjectiveRandomSearcher,
            )

            return MultiObjectiveRandomSearcher
        return RandomSearcher
    elif searcher_name == "bore":
        if is_multi_objective:
            raise ValueError(f"Searcher '{searcher_name}' does not support multi-objective optimization.")
        from syne_tune.optimizer.schedulers.searchers.bore import Bore

        return Bore
    elif searcher_name == "kde":
        if is_multi_objective:
            raise ValueError(f"Searcher '{searcher_name}' does not support multi-objective optimization.")
        from syne_tune.optimizer.schedulers.searchers.kde import (
            KernelDensityEstimator,
        )

        return KernelDensityEstimator
    elif searcher_name == "regularized_evolution":
        if is_multi_objective:
            from syne_tune.optimizer.schedulers.multiobjective.multi_objective_regularized_evolution import (
                MultiObjectiveRegularizedEvolution,
            )

            return MultiObjectiveRegularizedEvolution
        return RegularizedEvolution
    elif searcher_name == "cqr":
        if is_multi_objective:
            raise ValueError(f"Searcher '{searcher_name}' does not support multi-objective optimization.")
        from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import (
            ConformalQuantileRegression,
        )

        return ConformalQuantileRegression
    elif searcher_name == "botorch":
        if is_multi_objective:
            raise ValueError(f"Searcher '{searcher_name}' does not support multi-objective optimization.")
        from syne_tune.optimizer.schedulers.searchers.botorch.botorch_searcher import (
            BoTorchSearcher,
        )

        return BoTorchSearcher
    else:
        raise ValueError(f"Unknown searcher: {searcher_name}")


def searcher_factory(
    searcher_name: str, config_space: dict[str, Any], is_multi_objective: bool = False, **searcher_kwargs
) -> BaseSearcher:
    return searcher_cls(searcher_name, is_multi_objective=is_multi_objective)(config_space=config_space, **searcher_kwargs)
