import logging

from syne_tune.optimizer.schedulers.legacy_fifo import LegacyFIFOScheduler
from syne_tune.optimizer.schedulers.legacy_hyperband import LegacyHyperbandScheduler
from syne_tune.optimizer.schedulers.legacy_median_stopping_rule import (
    LegacyMedianStoppingRule,
)
from syne_tune.optimizer.schedulers.pbt import PopulationBasedTraining

__all__ = [
    "LegacyFIFOScheduler",
    "LegacyHyperbandScheduler",
    "LegacyMedianStoppingRule",
    "PopulationBasedTraining",
]

try:
    from syne_tune.optimizer.schedulers.ray_scheduler import (  # noqa: F401
        RayTuneScheduler,
    )

    __all__.append("RayTuneScheduler")
except ImportError as e:
    logging.debug(e)
