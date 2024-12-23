__all__ = []

import logging

try:
    from syne_tune.optimizer.schedulers.searchers.botorch.legacy_botorch_searcher import (  # noqa: F401
        LegacyBoTorchSearcher,
    )

    __all__.append("BoTorchSearcher")
    __all__.append("BotorchSearcher")  # deprecated
except ImportError as e:
    logging.debug(e)
