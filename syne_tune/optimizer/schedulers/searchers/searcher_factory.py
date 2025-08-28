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


def searcher_cls(searcher_name: str):
    """
    :param searcher_name:
    :return: the class associated to the searcher string
    """
    if searcher_name == "random_search":
        return RandomSearcher
    elif searcher_name == "bore":
        from syne_tune.optimizer.schedulers.searchers.bore import Bore

        return Bore
    elif searcher_name == "kde":
        from syne_tune.optimizer.schedulers.searchers.kde import (
            KernelDensityEstimator,
        )

        return KernelDensityEstimator
    elif searcher_name == "regularized_evolution":
        return RegularizedEvolution
    elif searcher_name == "cqr":
        from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import (
            ConformalQuantileRegression,
        )

        return ConformalQuantileRegression
    elif searcher_name == "botorch":
        from syne_tune.optimizer.schedulers.searchers.botorch.botorch_searcher import (
            BoTorchSearcher,
        )

        return BoTorchSearcher
    else:
        raise ValueError(f"Unknown searcher: {searcher_name}")


def searcher_factory(
    searcher_name: str, config_space: dict[str, Any], **searcher_kwargs
) -> BaseSearcher:
    return searcher_cls(searcher_name)(config_space=config_space, **searcher_kwargs)
