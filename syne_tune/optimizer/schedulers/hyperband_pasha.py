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
import numpy as np
from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem


class PASHARungSystem(PromotionRungSystem):
    """
    Implements PASHA algorithm. It is very similar to ASHA, but it progressively
    extends the maximum resources if the ranking in the top two current rungs changes.

    A report introducing and evaluating the approach is available at
    TODO: add link
    """

    def __init__(
        self,
        rung_levels,
        promote_quantiles,
        metric,
        mode,
        resource_attr,
        max_t,
        ranking_criterion,
        epsilon,
        epsilon_scaling,
    ):
        super().__init__(
            rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
        )
        self.ranking_criterion = ranking_criterion
        # define the index of the current top rung, starting from 1 for the lowest rung
        #
        self.current_rung_idx = 2
        self.rung_levels = rung_levels

        # initialize current maximum resources
        self.current_max_t = rung_levels[self.current_rung_idx - 1]

        self.epsilon = epsilon
        self.epsilon_scaling = epsilon_scaling

    # overriding the method in HB promotion to accomodate the increasing max resources level
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
            if rung.data != {}:
                trial_ids = rung.data.keys()
                values = []
                for trial_id in trial_ids:
                    values.append(rung.data[trial_id][0])
                # order specifies where the value should be placed in the sorted list
                values_order = np.array(values).argsort()
                # calling argsort on the order will give us the ranking
                values_ranking = values_order.argsort()
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
        if self._mode == "max":
            reverse = True
        else:
            reverse = False

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
        elif self.ranking_criterion == "soft_ranking_std":
            epsilon = (
                np.std([e[2] for e in sorted_previous_rung]) * self.epsilon_scaling
            )
        elif (
            self.ranking_criterion == "soft_ranking_median_dst"
            or self.ranking_criterion == "soft_ranking_mean_dst"
        ):
            scores = [e[2] for e in sorted_previous_rung]
            distances = [
                abs(e1 - e2)
                for idx1, e1 in enumerate(scores)
                for idx2, e2 in enumerate(scores)
                if idx1 != idx2
            ]
            if self.ranking_criterion == "soft_ranking_mean_dst":
                epsilon = np.mean(distances) * self.epsilon_scaling
            elif self.ranking_criterion == "soft_ranking_median_dst":
                epsilon = np.median(distances) * self.epsilon_scaling
            else:
                raise ValueError(
                    "Ranking criterion {} is not supported".format(
                        self.ranking_criterion
                    )
                )
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

    def on_task_report(self, trial_id: str, result: dict, skip_rungs: int) -> dict:
        """
        Apart from calling the superclass method, we also check the rankings
        and decides if to increase the current maximum resources.
        """
        ret_dict = super().on_task_report(trial_id, result, skip_rungs)

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
                self.current_max_t = self.max_t

        return ret_dict
