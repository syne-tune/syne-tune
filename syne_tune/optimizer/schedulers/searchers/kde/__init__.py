import logging

__all__ = []

try:
    from syne_tune.optimizer.schedulers.searchers.kde.kde_searcher import (  # noqa: F401
        KernelDensityEstimator,
    )

    __all__.extend(
        [
            "KernelDensityEstimator",
        ]
    )
except ImportError as e:
    logging.debug(e)
