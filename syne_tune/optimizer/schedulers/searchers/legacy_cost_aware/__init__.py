__all__ = []

import logging

try:
    from syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.cost_aware_gp_fifo_searcher import (  # noqa: F401
        CostAwareGPFIFOSearcher,
    )
    from syne_tune.optimizer.schedulers.searchers.legacy_cost_aware.cost_aware_gp_multifidelity_searcher import (  # noqa: F401
        CostAwareGPMultiFidelitySearcher,
    )

    __all__.extend(
        [
            "CostAwareGPFIFOSearcher",
            "CostAwareGPMultiFidelitySearcher",
        ]
    )
except ImportError as e:
    logging.debug(e)
