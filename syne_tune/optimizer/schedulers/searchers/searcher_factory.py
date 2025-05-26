from typing import Dict, Any

from syne_tune.optimizer.schedulers.searchers.bore import Bore
from syne_tune.optimizer.schedulers.searchers.botorch.botorch_searcher import (
    BoTorchSearcher,
)
from syne_tune.optimizer.schedulers.searchers.kde import KernelDensityEstimator
from syne_tune.optimizer.schedulers.searchers.random_searcher import RandomSearcher
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)
from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import (
    ConformalQuantileRegression,
)

searcher_dict = {
    "random_search": RandomSearcher,
    "bore": Bore,
    "kde": KernelDensityEstimator,
    "regularized_evolution": RegularizedEvolution,
    "cqr": ConformalQuantileRegression,
    "botorch": BoTorchSearcher,
}


def searcher_factory(
    searcher_name: str, config_space: Dict[str, Any], **searcher_kwargs
) -> BaseSearcher:
    assert (
        searcher_name in searcher_dict
    ), f"Searcher name must be one of {list(searcher_dict.keys())}, got {searcher_name}"
    return searcher_dict[searcher_name](config_space=config_space, **searcher_kwargs)
