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
from typing import Optional, Dict, Any, List, Union, Set

from syne_tune.config_space import config_space_size
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.utils import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Configuration,
    ConfigurationFilter,
)


class ExclusionList:
    """
    Maintains exclusion list of configs, to avoid choosing configs several
    times. In fact, ``self.excl_set`` maintains a set of match strings.

    The exclusion list contains non-extended configs, but it can be fed with
    and queried with extended configs. In that case, the resource attribute
    is removed from the config.

    :param hp_ranges: Encodes configurations to vectors
    :param configurations: Initial configurations. Default is empty
    """

    def __init__(
        self,
        hp_ranges: HyperparameterRanges,
        configurations: Optional[Union[List[Configuration], Set[str]]] = None,
    ):
        self.hp_ranges = hp_ranges
        keys = self.hp_ranges.internal_keys
        # Remove resource attribute from ``self.keys`` if present
        resource_attr = self.hp_ranges.name_last_pos
        if resource_attr is None:
            self.keys = keys
        else:
            pos = keys.index(resource_attr)
            self.keys = keys[:pos] + keys[(pos + 1) :]
        self.configspace_size = config_space_size(self.hp_ranges.config_space)
        if configurations is None:
            configurations = []
        if isinstance(configurations, list):
            self.excl_set = set(self._to_matchstr(config) for config in configurations)
        else:
            # Copy constructor
            assert isinstance(configurations, set)
            self.excl_set = configurations

    def _to_matchstr(self, config) -> str:
        return self.hp_ranges.config_to_match_string(config, keys=self.keys)

    def contains(self, config: Configuration) -> bool:
        return self._to_matchstr(config) in self.excl_set

    def add(self, config: Configuration):
        self.excl_set.add(self._to_matchstr(config))

    def copy(self) -> "ExclusionList":
        return ExclusionList(
            hp_ranges=self.hp_ranges,
            configurations=self.excl_set.copy(),
        )

    def __len__(self) -> int:
        return len(self.excl_set)

    def config_space_exhausted(self) -> bool:
        return (self.configspace_size is not None) and len(
            self.excl_set
        ) >= self.configspace_size

    def get_state(self) -> Dict[str, Any]:
        return {
            "excl_set": list(self.excl_set),
            "keys": self.keys,
        }

    def clone_from_state(self, state: Dict[str, Any]):
        self.keys = state["keys"]
        self.excl_set = set(state["excl_set"])


class ExclusionListFromState(ExclusionList):
    def __init__(
        self,
        state: TuningJobState,
        filter_observed_data: Optional[ConfigurationFilter] = None,
    ):
        super().__init__(
            hp_ranges=state.hp_ranges,
            configurations=state.all_configurations(filter_observed_data),
        )
