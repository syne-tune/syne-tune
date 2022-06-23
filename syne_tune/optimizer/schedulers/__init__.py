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

__all__ = [
    "FIFOScheduler",
    "HyperbandScheduler",
    "MedianStoppingRule",
    "PopulationBasedTraining",
]

import logging

from syne_tune.try_import import try_import_raytune_message
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.median_stopping_rule import MedianStoppingRule
from syne_tune.optimizer.schedulers.pbt import PopulationBasedTraining

try:
    from syne_tune.optimizer.schedulers.ray_scheduler import (  # noqa: F401
        RayTuneScheduler,
    )

    __all__.append("RayTuneScheduler")
except ImportError:
    logging.info(try_import_raytune_message())
