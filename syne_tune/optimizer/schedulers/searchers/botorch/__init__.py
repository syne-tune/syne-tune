
__all__ = []

import logging

try:
    from syne_tune.optimizer.schedulers.searchers.botorch.botorch_searcher import (  # noqa: F401
        BoTorchSearcher,
        BotorchSearcher,  # deprecated
    )

    __all__.append("BoTorchSearcher")
    __all__.append("BotorchSearcher")  # deprecated
except ImportError as e:
    logging.debug(e)
