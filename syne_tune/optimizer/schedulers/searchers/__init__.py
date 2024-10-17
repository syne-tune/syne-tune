import logging

from syne_tune.optimizer.schedulers.searchers.searcher import (  # noqa: F401
    BaseSearcher,
    impute_points_to_evaluate,
)
from syne_tune.optimizer.schedulers.searchers.searcher_base import (  # noqa: F401
    StochasticSearcher,
    StochasticAndFilterDuplicatesSearcher,
    extract_random_seed,
)
from syne_tune.optimizer.schedulers.searchers.random_grid_searcher import (  # noqa: F401
    RandomSearcher,
    GridSearcher,
)
from syne_tune.optimizer.schedulers.searchers.searcher_factory import (  # noqa: F401
    searcher_factory,
)

__all__ = [
    "BaseSearcher",
    "impute_points_to_evaluate",
    "StochasticSearcher",
    "StochasticAndFilterDuplicatesSearcher",
    "extract_random_seed",
    "RandomSearcher",
    "GridSearcher",
    "searcher_factory",
]

try:
    from syne_tune.optimizer.schedulers.searchers.model_based_searcher import (  # noqa: F401
        ModelBasedSearcher,
        BayesianOptimizationSearcher,
    )
    from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import (  # noqa: F401
        GPFIFOSearcher,
    )
    from syne_tune.optimizer.schedulers.searchers.gp_multifidelity_searcher import (  # noqa: F401
        GPMultiFidelitySearcher,
    )

    __all__.extend(
        [
            "ModelBasedSearcher",
            "BayesianOptimizationSearcher",
            "GPFIFOSearcher",
            "GPMultiFidelitySearcher",
        ]
    )
except ImportError as e:
    logging.debug(e)
