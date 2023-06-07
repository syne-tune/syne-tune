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
import itertools
from typing import Any, Dict, List

import numpy as np
from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem


class PASHARungSystem(PromotionRungSystem):
    """
    Implements PASHA algorithm. PASHA is a more efficient version of ASHA
    and is able to dynamically allocate maximum resources for the tuning procedure
    depending on the need. Experimental evaluation has shown PASHA consumes
    significantly fewer computational resources than ASHA.

    For more details, see the paper:
        | Bohdal, Balles, Wistuba, Ermis, Archambeau, Zappella (2023)
        | PASHA: Efficient HPO and NAS with Progressive Resource Allocation
        | https://openreview.net/forum?id=syfgJE6nFRW
    """

    def __init__(
        self,
        rung_levels: List[int],
        promote_quantiles: List[float],
        metric: str,
        mode: str,
        resource_attr: str,
        max_t: int,
    ):
        super().__init__(
            rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
        )
        # define the index of the current top rung, starting from 1 for the
        # lowest rung
        assert self.num_rungs >= 1, "rung_levels must not be empty"
        self.rung_levels = rung_levels.copy()

        # initialize current maximum resources
        self.current_rung_idx = min(len(rung_levels) - 1, 2)
        self.current_max_t = rung_levels[self.current_rung_idx - 1]

        self.epsilon = 0.0
        self.per_epoch_results = {}
        self.epoch_to_trials = {}
        self.current_max_epoch = -1

    # overriding the method in HB promotion to accommodate the increasing max
    # resources level
    def _effective_max_t(self):
        return self.current_max_t

    def _get_top_two_rungs_rankings(self):
        """
        Look at the current top two rungs and get the rankings of the configurations.
        The rungs can be empty, in which case we will return a list with 0 or 1 elements.
        Normally the list will include rankings for both rungs.

        The rankings are stored as a list of tuples (trial_id, rank, value).

        Lower values have lower ranks, starting from zero. For example:
        [('0', 0, 10.0), ('1', 3, 19.6), ('2', 2, 14.3), ('3', 1, 11.6)]

        :return: rankings
            List of at most two lists with tuple(trial_id, rank, score)
        """
        rankings = []
        # be careful, self._rungs is ordered with the highest resources level in the beginning
        for rung in [
            self._rungs[-self.current_rung_idx],
            self._rungs[-self.current_rung_idx + 1],
        ]:
            if rung:
                # Note that entries in ``rung.data`` are already sorted
                trial_ids, values = zip(
                    *[(x.trial_id, x.metric_val) for x in rung.data]
                )
                # Note: To retain the exact logic of older code (which used
                # ``numpy.argsort``, we assume the ranks for increasing metric
                # values, which is the opposite ordering to ``rung.data`` if
                # ``self._mode == "max"``
                if self._mode == "min":
                    values_ranking = list(range(len(trial_ids)))
                else:
                    values_ranking = list(range(len(trial_ids) - 1, -1, -1))
                ranking = list(zip(trial_ids, values_ranking, values))
                rankings.append(ranking)
        return rankings

    def _get_sorted_top_rungs(self, rankings):
        """
        Sort the configurations in the top rung and the previous rung.
        Filter out the configurations from the previous rung that
        are not in the top rung.

        :param rankings: list of at most two lists with tuple(trial_id, rank, score)
        return: sorted_top_rung, sorted_previous_rung
        """
        # filter only the relevant configurations from the earlier rung
        top_rung_keys = set([e[0] for e in rankings[0]])
        corresponding_previous_rung_trials = filter(
            lambda e: e[0] in top_rung_keys, rankings[1]
        )
        # if we try to maximize the objective, we need to reverse the ranking
        reverse = self._mode == "max"
        sorted_top_rung = sorted(rankings[0], key=lambda e: e[1], reverse=reverse)
        sorted_previous_rung = sorted(
            corresponding_previous_rung_trials, key=lambda e: e[1], reverse=reverse
        )
        return sorted_top_rung, sorted_previous_rung

    def _evaluate_soft_ranking(self, sorted_top_rung, sorted_previous_rung) -> bool:
        """
        Soft ranking creates groups of similarly performing configurations
        and increases the resources only if a configuration goes outside of
        its group.

        :param sorted_top_rung: list of tuple(trial_id, rank, score)
        :param sorted_previous_rung: list of tuple(sorted_top_rung, rank, score)
        :return: keep_current_budget
        """
        keep_current_budget = True
        if len(sorted_previous_rung) < 2:
            epsilon = 0.0
        else:
            epsilon = self.epsilon

        # create groups of configurations with similar performance
        previous_rung_groups = []
        for idx, item in enumerate(sorted_previous_rung):
            current_rung_group = [item[0]]
            # add configurations that are after the current configuration
            for idx_after in range(idx + 1, len(sorted_previous_rung)):
                new_item = sorted_previous_rung[idx_after]

                if self._mode == "max":
                    if new_item[2] < item[2] - epsilon:
                        break
                else:
                    if new_item[2] > item[2] + epsilon:
                        break
                current_rung_group.append(new_item[0])
            # add configurations that are before the current configuration
            for idx_before in range(idx - 1, -1, -1):
                new_item = sorted_previous_rung[idx_before]
                if self._mode == "max":
                    if new_item[2] > item[2] + epsilon:
                        break
                else:
                    if new_item[2] < item[2] - epsilon:
                        break
                current_rung_group.append(new_item[0])
            previous_rung_groups.append(set(current_rung_group))

        # evaluate if a configuration has switched its group
        for idx, item in enumerate(sorted_top_rung):
            if item[0] not in previous_rung_groups[idx]:
                keep_current_budget = False
                break

        return keep_current_budget

    def _update_epsilon(self):
        """
        This function is used to automatically calculate the value of epsilon.
        It finds the configurations which swapped their rankings across rungs
        and estimates the value of epsilon as the 90th percentile of the difference
        between their performance in the previous rung.

        The original value of epsilon is kept if no suitable configurations were found.
        """

        seen_pairs = set()
        noisy_cfg_distances = []
        top_epoch = min(
            self.current_max_epoch, self._rungs[-self.current_rung_idx].level
        )
        bottom_epoch = min(
            self._rungs[-self.current_rung_idx + 1].level, self.current_max_epoch
        )
        for epoch in range(top_epoch, bottom_epoch, -1):
            if len(self.epoch_to_trials[epoch]) > 1:
                for pair in itertools.combinations(self.epoch_to_trials[epoch], 2):
                    c1, c2 = pair[0], pair[1]
                    if (c1, c2) not in seen_pairs:
                        seen_pairs.add((c1, c2))
                        p1, p2 = (
                            self.per_epoch_results[c1][epoch],
                            self.per_epoch_results[c2][epoch],
                        )
                        cond = p1 > p2

                        opposite_order = False
                        same_order_after_opposite = False
                        # now we need to check the earlier epochs to see if at any point they had a different order
                        for prev_epoch in range(epoch - 1, 0, -1):
                            pp1, pp2 = (
                                self.per_epoch_results[c1][prev_epoch],
                                self.per_epoch_results[c2][prev_epoch],
                            )
                            p_cond = pp1 > pp2
                            if p_cond == (not cond):
                                opposite_order = True
                            if opposite_order and p_cond == cond:
                                same_order_after_opposite = True
                                break

                        if opposite_order and same_order_after_opposite:
                            noisy_cfg_distances.append(abs(p1 - p2))

        if len(noisy_cfg_distances) > 0:
            self.epsilon = np.percentile(noisy_cfg_distances, 90)
            if str(self.epsilon) == "nan":
                raise ValueError("Epsilon became nan")

    def _update_per_epoch_results(self, trial_id, result):
        resource = result[self._resource_attr]
        metric_val = result[self._metric]
        if trial_id not in self.per_epoch_results:
            self.per_epoch_results[trial_id] = dict()
        self.per_epoch_results[trial_id][resource] = metric_val
        if resource not in self.epoch_to_trials:
            self.epoch_to_trials[resource] = set()
        self.epoch_to_trials[resource].add(trial_id)
        self.current_max_epoch = max(self.current_max_epoch, resource)

    def _decide_resource_increase(self, rankings) -> bool:
        """
        Decide if to increase the resources given the current rankings.
        Currently we look at the rankings and if elements in the first list
        have the same order also in the second list, we keep the current resource
        budget. If the rankings are different, we will increase the budget.

        The rankings can only be incorrect if we have rankings for both rungs.

        :param rankings: list of at most two lists with tuple(trial_id, rank, score)
        :return: not keep_current_budget
        """
        if len(rankings) == 2:
            sorted_top_rung, sorted_previous_rung = self._get_sorted_top_rungs(rankings)
        else:
            return False

        keep_current_budget = self._evaluate_soft_ranking(
            sorted_top_rung, sorted_previous_rung
        )

        return not keep_current_budget

    def on_task_report(
        self, trial_id: str, result: Dict[str, Any], skip_rungs: int
    ) -> Dict[str, Any]:
        """
        Apart from calling the superclass method, we also check the rankings
        and decides if to increase the current maximum resources.
        """
        ret_dict = super().on_task_report(trial_id, result, skip_rungs)

        self._update_per_epoch_results(trial_id, result)
        self._update_epsilon()

        # check the rankings and decide if to increase the current maximum resources
        rankings = self._get_top_two_rungs_rankings()
        increase_resources = self._decide_resource_increase(rankings)

        # we have a maximum amount of resources that PASHA can use
        # the resources should not increase indefinitely
        if increase_resources:
            if self.current_rung_idx < len(self._rungs):
                self.current_rung_idx += 1
                # be careful, self.rung_levels is ordered with the highest resources level at the end
                # moreover, since we use rung levels for counting both from the beginning and from the end of the list
                # we need to remember that counting from the beginning it's zero indexed
                self.current_max_t = self.rung_levels[self.current_rung_idx - 1]
            else:
                self.current_max_t = self._max_t

        return ret_dict
