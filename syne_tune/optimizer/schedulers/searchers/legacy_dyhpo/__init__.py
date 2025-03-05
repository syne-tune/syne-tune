__all__ = []

import logging

try:
    from syne_tune.optimizer.schedulers.searchers.legacy_dyhpo.dyhpo_searcher import (  # noqa: F401
        DynamicHPOSearcher,
    )

    __all__.append("DynamicHPOSearcher")
except ImportError as e:
    logging.debug(e)
