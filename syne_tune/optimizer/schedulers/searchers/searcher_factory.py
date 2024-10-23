from typing import Dict, Any

from syne_tune.optimizer.schedulers.searchers.bore import Bore
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import GPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.kde import KernelDensityEstimator
from syne_tune.optimizer.schedulers.searchers.random_grid_searcher import RandomSearcher
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher

searcher_cls_dict = {
    "random_search": RandomSearcher,
    "bore": Bore,
    "kde": KernelDensityEstimator,
    "bayesopt": GPFIFOSearcher,
}


def searcher_factory(
    searcher_name: str, config_space: Dict[str, Any], **searcher_kwargs
) -> BaseSearcher:
    assert searcher_name in searcher_cls_dict
    return searcher_cls_dict[searcher_name](
        config_space=config_space, **searcher_kwargs
    )
