# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import logging

from syne_tune.try_import import try_import_gpsearchers_message

from syne_tune.optimizer.schedulers.searchers.searcher import (  # noqa: F401
    BaseSearcher,
    impute_points_to_evaluate,
)
from syne_tune.optimizer.schedulers.searchers.searcher_base import (  # noqa: F401
    StochasticSearcher,
    StochasticAndFilterDuplicatesSearcher,
    extract_random_seed,
)
from syne_tune.optimizer.schedulers.searchers.random_grid_searcher import (  # noqa: F401
    RandomSearcher,
    GridSearcher,
)
from syne_tune.optimizer.schedulers.searchers.searcher_factory import (  # noqa: F401
    searcher_factory,
)
from syne_tune.optimizer.schedulers.searchers.model_based_searcher import (  # noqa: F401
    ModelBasedSearcher,
    BayesianOptimizationSearcher,
)

__all__ = [
    "BaseSearcher",
    "impute_points_to_evaluate",
    "StochasticSearcher",
    "StochasticAndFilterDuplicatesSearcher",
    "extract_random_seed",
    "RandomSearcher",
    "GridSearcher",
    "searcher_factory",
    "ModelBasedSearcher",
    "BayesianOptimizationSearcher",
]

try:
    from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import (  # noqa: F401
        GPFIFOSearcher,
    )
    from syne_tune.optimizer.schedulers.searchers.gp_multifidelity_searcher import (  # noqa: F401
        GPMultiFidelitySearcher,
    )

    __all__.extend(
        [
            "GPFIFOSearcher",
            "GPMultiFidelitySearcher",
        ]
    )
except ImportError:
    logging.info(try_import_gpsearchers_message())
