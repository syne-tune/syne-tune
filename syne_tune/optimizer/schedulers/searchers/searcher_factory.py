from typing import Dict, Any
from syne_tune.optimizer.schedulers.searchers.random_searcher import RandomSearcher
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)


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
    match searcher_name:
        case "random_search":
            cls = RandomSearcher
        case "bore":
            from syne_tune.optimizer.schedulers.searchers.bore import Bore

            cls = Bore
        case "kde":
            from syne_tune.optimizer.schedulers.searchers.kde import (
                KernelDensityEstimator,
            )

            cls = KernelDensityEstimator
        case "regularized_evolution":
            cls = RegularizedEvolution
        case "cqr":
            from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import (
                ConformalQuantileRegression,
            )

            cls = ConformalQuantileRegression
        case "botorch":
            from syne_tune.optimizer.schedulers.searchers.botorch.botorch_searcher import (
                BoTorchSearcher,
            )

            cls = BoTorchSearcher
        case _:
            raise ValueError(f"Unknown searcher: {searcher_name}")
    return cls


def searcher_factory(
    searcher_name: str, config_space: Dict[str, Any], **searcher_kwargs
) -> BaseSearcher:
    return searcher_cls(searcher_name)(config_space=config_space, **searcher_kwargs)
