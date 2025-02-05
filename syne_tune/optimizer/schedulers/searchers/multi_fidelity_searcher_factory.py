from typing import Dict, Any

from syne_tune.optimizer.schedulers.searchers.bore import MultiFidelityBore
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate_searcher import (
    SurrogateSearcher,
)
from syne_tune.optimizer.schedulers.searchers.kde import (
    MultiFidelityKernelDensityEstimator,
)
from syne_tune.optimizer.schedulers.searchers.random_searcher import RandomSearcher
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher

multi_fidelity_searcher_cls_dict = {
    "random_search": RandomSearcher,
    "bore": MultiFidelityBore,
    "kde": MultiFidelityKernelDensityEstimator,
    "cqr": SurrogateSearcher,
}


def multi_fidelity_searcher_factory(
    searcher_name: str, config_space: Dict[str, Any], **searcher_kwargs
) -> BaseSearcher:
    assert searcher_name in multi_fidelity_searcher_cls_dict
    return multi_fidelity_searcher_cls_dict[searcher_name](
        config_space=config_space, **searcher_kwargs
    )
