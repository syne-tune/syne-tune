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
from typing import Dict, Optional, List
import logging

from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_impl import (
    HyperparameterRangesImpl,
)

logger = logging.getLogger(__name__)


def make_hyperparameter_ranges(
    config_space: Dict,
    name_last_pos: Optional[str] = None,
    value_for_last_pos=None,
    active_config_space: Optional[Dict] = None,
    prefix_keys: Optional[List[str]] = None,
) -> HyperparameterRanges:
    hp_ranges = HyperparameterRangesImpl(
        config_space,
        name_last_pos=name_last_pos,
        value_for_last_pos=value_for_last_pos,
        active_config_space=active_config_space,
        prefix_keys=prefix_keys,
    )
    return hp_ranges
