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
from syne_tune.try_import import try_import_gpsearchers_message

__all__ = []

try:
    from syne_tune.optimizer.schedulers.searchers.cost_aware.cost_aware_gp_fifo_searcher import (  # noqa: F401
        CostAwareGPFIFOSearcher,
    )
    from syne_tune.optimizer.schedulers.searchers.cost_aware.cost_aware_gp_multifidelity_searcher import (  # noqa: F401
        CostAwareGPMultiFidelitySearcher,
    )

    __all__.extend(
        [
            "CostAwareGPFIFOSearcher",
            "CostAwareGPMultiFidelitySearcher",
        ]
    )
except ImportError:
    print(try_import_gpsearchers_message())
