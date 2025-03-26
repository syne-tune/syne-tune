import logging

from syne_tune.optimizer.schedulers.searchers.legacy_searcher import (  # noqa: F401
    LegacyBaseSearcher,
    impute_points_to_evaluate,
)
from syne_tune.optimizer.schedulers.searchers.searcher_base import (  # noqa: F401
    StochasticSearcher,
    StochasticAndFilterDuplicatesSearcher,
    extract_random_seed,
)
from syne_tune.optimizer.schedulers.searchers.legacy_random_grid_searcher import (  # noqa: F401
    LegacyRandomSearcher,
    GridSearcher,
)
from syne_tune.optimizer.schedulers.searchers.legacy_searcher_factory import (
    legacy_searcher_factory,
)

__all__ = [
    "LegacyBaseSearcher",
    "impute_points_to_evaluate",
    "StochasticSearcher",
    "StochasticAndFilterDuplicatesSearcher",
    "extract_random_seed",
    "LegacyRandomSearcher",
    "GridSearcher",
    "legacy_searcher_factory",
]

try:
    from syne_tune.optimizer.schedulers.searchers.legacy_model_based_searcher import (  # noqa: F401
        ModelBasedSearcher,
        BayesianOptimizationSearcher,
    )
    from syne_tune.optimizer.schedulers.searchers.legacy_gp_fifo_searcher import (  # noqa: F401
        GPFIFOSearcher,
    )
    from syne_tune.optimizer.schedulers.searchers.legacy_gp_multifidelity_searcher import (  # noqa: F401
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
