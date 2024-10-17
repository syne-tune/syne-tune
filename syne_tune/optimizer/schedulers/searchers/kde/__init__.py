import logging

__all__ = []

try:
    from syne_tune.optimizer.schedulers.searchers.kde.kde_searcher import (  # noqa: F401
        KernelDensityEstimator,
    )
    from syne_tune.optimizer.schedulers.searchers.kde.multi_fidelity_kde_searcher import (  # noqa: F401
        MultiFidelityKernelDensityEstimator,
    )

    __all__.extend(
        [
            "KernelDensityEstimator",
            "MultiFidelityKernelDensityEstimator",
        ]
    )
except ImportError as e:
    logging.debug(e)
