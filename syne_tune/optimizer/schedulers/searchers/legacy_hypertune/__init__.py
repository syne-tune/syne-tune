import logging

__all__ = []

try:
    from syne_tune.optimizer.schedulers.searchers.legacy_hypertune.hypertune_searcher import (  # noqa: F401
        HyperTuneSearcher,
    )

    __all__.append("HyperTuneSearcher")
except ImportError as e:
    logging.debug(e)
