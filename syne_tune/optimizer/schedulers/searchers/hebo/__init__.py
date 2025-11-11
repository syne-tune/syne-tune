import logging

__all__ = []

try:
    from syne_tune.optimizer.schedulers.searchers.hebo.hebo_searcher import (  # noqa: F401
        HEBOSearcher,
    )

    __all__.extend(
        [
            "HEBOSearcher",
        ]
    )
except ImportError as e:
    logging.warning(
        f"Installing Optuna is required to run the HEBOSearcher, please make sure you have it installed. \n Error:{e}"
    )
