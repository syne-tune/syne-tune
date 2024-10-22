import logging

from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from syne_tune.optimizer.schedulers.multiobjective.multi_objective_regularized_evolution import (
    MultiObjectiveRegularizedEvolution,
)
from syne_tune.optimizer.schedulers.multiobjective.nsga2_searcher import (
    NSGA2Searcher,
)
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)

__all__ = [
    "MOASHA",
    "MultiObjectiveRegularizedEvolution",
    "NSGA2Searcher",
    "LinearScalarizedScheduler",
]

try:
    from syne_tune.optimizer.schedulers.multiobjective.multi_surrogate_multi_objective_searcher import (  # noqa: F401
        MultiObjectiveMultiSurrogateSearcher,
    )
    from syne_tune.optimizer.schedulers.multiobjective.random_scalarization import (  # noqa: F401
        MultiObjectiveLCBRandomLinearScalarization,
    )

    __all__.extend(
        [
            "MultiObjectiveMultiSurrogateSearcher",
            "MultiObjectiveLCBRandomLinearScalarization",
        ]
    )
except ImportError as e:
    logging.debug(e)
