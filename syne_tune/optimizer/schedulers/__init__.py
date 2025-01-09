import logging

from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.median_stopping_rule import MedianStoppingRule
from syne_tune.optimizer.schedulers.legacy_pbt import PopulationBasedTraining

__all__ = [
    "FIFOScheduler",
    "HyperbandScheduler",
    "MedianStoppingRule",
    "PopulationBasedTraining",
]

try:
    from syne_tune.optimizer.schedulers.ray_scheduler import (  # noqa: F401
        RayTuneScheduler,
    )

    __all__.append("RayTuneScheduler")
except ImportError as e:
    logging.debug(e)
