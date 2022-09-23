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
from typing import Iterator, List, Union, Optional
import numpy as np
import logging

from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    CandidateGenerator,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Configuration,
    ConfigurationFilter,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.config_space import config_space_size

logger = logging.getLogger(__name__)


MAX_RETRIES_CANDIDATES_EN_BULK = 20


class RandomStatefulCandidateGenerator(CandidateGenerator):
    """
    This generator maintains a random state, so if generate_candidates is
    called several times, different sequences are returned.

    """

    def __init__(
        self, hp_ranges: HyperparameterRanges, random_state: np.random.RandomState
    ):
        self.hp_ranges = hp_ranges
        self.random_state = random_state

    def generate_candidates(self) -> Iterator[Configuration]:
        while True:
            yield self.hp_ranges.random_config(self.random_state)

    def generate_candidates_en_bulk(
        self, num_cands: int, exclusion_list=None
    ) -> List[Configuration]:
        if exclusion_list is None:
            return self.hp_ranges.random_configs(self.random_state, num_cands)
        else:
            assert isinstance(
                exclusion_list, ExclusionList
            ), "exclusion_list must be of type ExclusionList"
            configs = []
            num_done = 0
            for i in range(MAX_RETRIES_CANDIDATES_EN_BULK):
                # From iteration 1, we request more than what is still
                # needed, because the config space seems to not have
                # enough configs left
                num_requested = min(num_cands, (num_cands - num_done) * (i + 1))
                new_configs = [
                    config
                    for config in self.hp_ranges.random_configs(
                        self.random_state, num_requested
                    )
                    if not exclusion_list.contains(config)
                ]
                num_new = min(num_cands - num_done, len(new_configs))
                configs += new_configs[:num_new]
                num_done += num_new
                if num_done == num_cands:
                    break
            if num_done < num_cands:
                logger.warning(
                    f"Could only sample {num_done} candidates where "
                    f"{num_cands} were requested. len(exclusion_list) = "
                    f"{len(exclusion_list)}"
                )
            return configs


class ExclusionList:
    """
    Maintains exclusion list of configs, to avoid choosing configs several
    times.

    The exclusion list contains non-extended configs, but it can be fed with
    and queried with extended configs. In that case, the resource attribute
    is removed from the config.

    """

    def __init__(
        self,
        state: Union[TuningJobState, dict],
        filter_observed_data: Optional[ConfigurationFilter] = None,
    ):
        is_new = isinstance(state, TuningJobState)
        if is_new:
            self.hp_ranges = state.hp_ranges
            keys = self.hp_ranges.internal_keys
            # Remove resource attribute from `self.keys` if present
            resource_attr = self.hp_ranges.name_last_pos
            if resource_attr is None:
                self.keys = keys
            else:
                pos = keys.index(resource_attr)
                self.keys = keys[:pos] + keys[(pos + 1) :]
            _elist = [
                x.trial_id for x in state.pending_evaluations
            ] + state.failed_trials
            observed_trial_ids = [x.trial_id for x in state.trials_evaluations]
            if filter_observed_data is not None:
                observed_trial_ids = [
                    trial_id
                    for trial_id in observed_trial_ids
                    if filter_observed_data(state.config_for_trial[trial_id])
                ]
            _elist = set(_elist + observed_trial_ids)
            self.excl_set = set(
                self._to_matchstr(state.config_for_trial[trial_id])
                for trial_id in _elist
            )
        else:
            self.hp_ranges = state["hp_ranges"]
            self.excl_set = state["excl_set"]
            self.keys = state["keys"]
        self.configspace_size = config_space_size(self.hp_ranges.config_space)

    def _to_matchstr(self, config) -> str:
        return self.hp_ranges.config_to_match_string(config, keys=self.keys)

    def contains(self, config: Configuration) -> bool:
        return self._to_matchstr(config) in self.excl_set

    def add(self, config: Configuration):
        self.excl_set.add(self._to_matchstr(config))

    def copy(self) -> "ExclusionList":
        return ExclusionList(
            {
                "hp_ranges": self.hp_ranges,
                "excl_set": set(self.excl_set),
                "keys": self.keys,
            }
        )

    @staticmethod
    def empty_list(hp_ranges: HyperparameterRanges) -> "ExclusionList":
        return ExclusionList(TuningJobState.empty_state(hp_ranges))

    def __len__(self) -> int:
        return len(self.excl_set)

    def config_space_exhausted(self) -> bool:
        return (self.configspace_size is not None) and len(
            self.excl_set
        ) >= self.configspace_size


MAX_RETRIES_ON_DUPLICATES = 10000


def generate_unique_candidates(
    candidates_generator: CandidateGenerator,
    num_candidates: int,
    exclusion_candidates: ExclusionList,
) -> List[Configuration]:
    exclusion_candidates = exclusion_candidates.copy()  # Copy
    result = []
    num_results = 0
    retries = 0
    just_added = True
    for i, cand in enumerate(candidates_generator.generate_candidates()):
        if just_added:
            if exclusion_candidates.config_space_exhausted():
                logger.warning(
                    "All entries of finite config space (size "
                    + f"{exclusion_candidates.configspace_size}) have been selected. Returning "
                    + f"{len(result)} configs instead of {num_candidates}"
                )
                break
            just_added = False
        if not exclusion_candidates.contains(cand):
            result.append(cand)
            num_results += 1
            exclusion_candidates.add(cand)
            retries = 0
            just_added = True
        else:
            # found a duplicate; retry
            retries += 1
        # End loop if enough candidates where generated, or after too many retries
        # (this latter can happen when most of them are duplicates, and must be done
        # to avoid infinite loops in the purely discrete case)
        if num_results == num_candidates or retries > MAX_RETRIES_ON_DUPLICATES:
            if retries > MAX_RETRIES_ON_DUPLICATES:
                logger.warning(
                    f"Reached limit of {MAX_RETRIES_ON_DUPLICATES} retries "
                    f"with i={i}. Returning {len(result)} candidates instead "
                    f"of the requested {num_candidates}"
                )
            break
    return result
