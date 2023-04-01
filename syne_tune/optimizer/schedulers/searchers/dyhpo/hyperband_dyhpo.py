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
from typing import List, Dict, Any, Tuple
from numpy.random import RandomState
import logging
from collections import Counter

from syne_tune.optimizer.schedulers.hyperband_promotion import (
    PromotionRungSystem,
)
from syne_tune.optimizer.schedulers.searchers.dyhpo.dyhpo_searcher import (
    DynamicHPOSearcher,
    KEY_NEW_CONFIGURATION,
)

logger = logging.getLogger(__name__)


DEFAULT_SH_PROBABILITY = 0.25


class ScheduleDecision:
    PROMOTE_SH = 0
    PROMOTE_DYHPO = 1
    START_DYHPO = 2


_SUMMARY_SCHEDULE_RECORDS = [
    ("promoted_by_sh", ScheduleDecision.PROMOTE_SH),
    ("promoted_by_dyhpo", ScheduleDecision.PROMOTE_DYHPO),
    ("started_by_dyhpo", ScheduleDecision.START_DYHPO),
]


class DyHPORungSystem(PromotionRungSystem):
    """
    Implements the logic which decides which paused trial to promote to the
    next resource level, or alternatively which configuration to start as a
    new trial, proposed in:

        | Wistuba, M. and Kadra, A. and Grabocka, J.
        | Dynamic and Efficient Gray-Box Hyperparameter Optimization for Deep Learning
        | https://arxiv.org/abs/2202.09774

    We do promotion-based scheduling, as in
    :class:`~syne_tune.optimizer.schedulers.hyperband_promotion.PromotionRungSystem`.
    In fact, we run the successive halving rule in :meth:`on_task_schedule` with
    probability ``probability_sh``, and the DyHPO logic otherwise, or if the SH
    rule does not promote a trial. This mechanism (not contained in the paper)
    ensures that trials are promoted eventually, even if DyHPO only starts new
    trials.

    Since :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` was designed
    for promotion decisions to be separate from decisions about new configs, the
    overall workflow is a bit tricky:

    * In :meth:`FIFOScheduler._suggest`, we first call
      :code:`promote_trial_id, extra_kwargs = self._promote_trial()`. If
      ``promote_trial_id != None``, this trial is promoted. Otherwise, we call
      :code:`config = self.searcher.get_config(**extra_kwargs, trial_id=trial_id)`
      and start a new trial with this config. In most cases, :meth:`_promote_trial`
      makes a promotion decision without using the searcher.
    * Here, we use the fact that information can be passed from
      :meth:`_promote_trial` to ``self.searcher.get_config`` via ``extra_kwargs``.
      Namely, :meth:``HyperbandScheduler._promote_trial` calls
      :meth:`on_task_schedule` here, which calls
      :meth:`~syne_tune.optimizer.schedulers.searchers.dyhpo.DynamicHPOSearcher.score_paused_trials_and_new_configs`,
      where everything happens.
    * First, all paused trials are scored w.r.t. the value of running them for one
      more unit of resource. Also, a number of random configs are scored w.r.t.
      the value of running them to the minimum resource.
    * If the winning config is from a paused trial, this is resumed. If the
      winning config is a new one, :meth:`on_task_schedule` returns this
      config using a special key :const:`KEY_NEW_CONFIGURATION`. This dict
      becomes part of ``extra_kwargs`` and is passed to ``self.searcher.get_config``
    * :meth:`~syne_tune.optimizer.schedulers.searchers.dyhpo.DynamicHPOSearcher.get_config`
      is trivial. It obtains an argument of name :const:`KEY_NEW_CONFIGURATION`
      returns its value, which is the winning config to be started as new trial

    We can ignore ``rung_levels`` and ``promote_quantiles``, they are not used.
    For each trial, we only need to maintain the resource level at which it is
    paused.
    """

    def __init__(
        self,
        rung_levels: List[int],
        promote_quantiles: List[float],
        metric: str,
        mode: str,
        resource_attr: str,
        max_t: int,
        searcher: DynamicHPOSearcher,
        probability_sh: bool,
        random_state: RandomState,
    ):
        assert len(rung_levels) > 0, "rung_levels must not be empty"
        assert isinstance(
            searcher, DynamicHPOSearcher
        ), "searcher must be of type DynamicHPOSearcher. Use searcher='dyhpo'"
        assert (
            0 <= probability_sh < 1
        ), f"probability_sh = {probability_sh}, must be in [0, 1)"
        super().__init__(
            rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
        )
        self._check_rung_levels(rung_levels)
        self._searcher = searcher
        self._min_resource = rung_levels[0]
        self._probability_sh = probability_sh
        self._random_state = random_state
        # Maps rung level to the one below
        self._previous_rung_level = dict(zip(rung_levels[1:] + [max_t], rung_levels))
        # Keeps a record of outcomes of :meth:`on_task_schedule` calls. Entries
        # are ``(trial_id, decision, milestone)``, where ``decision`` is
        # constant from :class:`ScheduleDecision`, and ``milestone`` is the
        # rung level which the trial reaches next
        self._schedule_records = []

    @staticmethod
    def _check_rung_levels(rung_levels: List[int]):
        if len(rung_levels) > 1:
            rmin = rung_levels[0]
            step = rung_levels[1] - rmin
            should_be = list(range(rmin, rung_levels[-1] + 1, step))
            if rmin != step or rung_levels != should_be:
                logger.warning(
                    "DyHPO should be run with linearly spaced rung levels, in "
                    "that reduction_factor is not used, and grace_period == "
                    "rung_increment, bracket == 1. Running with rung_levels = "
                    f"{rung_levels} is not recommended"
                )

    def _paused_trials_and_milestones(self) -> List[Tuple[str, int, int]]:
        """
        Return list of all trials which are paused. Entries are
        ``(trial_id, pos, resource)``, where ``pos`` is the position of the trial
        in its rung, and ``resource`` is the next rung level the trial reaches
        after being resumed.

        :return: See above
        """
        paused_trials = []
        next_level = self._max_t
        for rung in self._rungs:
            level = rung.level
            paused_trials.extend(
                (entry.trial_id, pos, next_level)
                for pos, entry in enumerate(rung.data)
                if self._is_promotable_trial(entry, level)
            )
            next_level = level
        return paused_trials

    def on_task_schedule(self, new_trial_id: str) -> Dict[str, Any]:
        """
        The main decision making happens here. We collect ``(trial_id, resource)``
        for all paused trials and call ``searcher``. The searcher scores all
        these trials along with a certain number of randomly drawn new
        configurations.

        If one of the paused trials has the best score, we return its ``trial_id``
        along with extra information, so it gets promoted.
        If one of the new configurations has the best score, we return this
        configuration. In this case, a new trial is started with this configuration.

        Note: For this scheduler type, ``kwargs`` must contain the trial ID of
        the new trial to be started, in case none can be promoted.
        """
        if self._random_state.rand() <= self._probability_sh:
            # Try to promote trial based on successive halving logic
            result = super().on_task_schedule(new_trial_id)
            if result.get("trial_id") is not None:
                self._schedule_records.append(
                    (
                        result["trial_id"],
                        ScheduleDecision.PROMOTE_SH,
                        result["milestone"],
                    )
                )
                return result
        # Follow DyHPO logic
        paused_trials = self._paused_trials_and_milestones()
        assert new_trial_id is not None, (
            "Internal error: kwargs must contain 'trial_id', the ID for a new "
            "trial to be started if no paused one is resumed. Make sure to "
            "pass this to the _promote_trial method when calling it in "
            "_suggest"
        )
        result = self._searcher.score_paused_trials_and_new_configs(
            paused_trials,
            min_resource=self._min_resource,
            new_trial_id=new_trial_id,
        )
        trial_id = result.get("trial_id")
        if trial_id is not None:
            # Trial is to be promoted
            pos = result["pos"]  # Position of trial in its rung
            milestone = next(r for i, _, r in paused_trials if i == trial_id)
            resume_from = self._previous_rung_level[milestone]
            rung = next(rung for rung in self._rungs if rung.level == resume_from)
            self._mark_as_promoted(rung, pos, trial_id=trial_id)
            ret_dict = {
                "trial_id": trial_id,
                "resume_from": resume_from,
                "milestone": milestone,
            }
            self._schedule_records.append(
                (trial_id, ScheduleDecision.PROMOTE_DYHPO, milestone)
            )
        else:
            # New trial is to be started
            ret_dict = {KEY_NEW_CONFIGURATION: result["config"]}
            self._schedule_records.append(
                (new_trial_id, ScheduleDecision.START_DYHPO, self._min_resource)
            )
        return ret_dict

    @property
    def schedule_records(self) -> List[Tuple[str, int, int]]:
        return self._schedule_records

    @staticmethod
    def summary_schedule_keys() -> List[str]:
        return [key for key, _ in _SUMMARY_SCHEDULE_RECORDS]

    def summary_schedule_records(self) -> Dict[str, Any]:
        histogram = Counter([x[1] for x in self._schedule_records])
        return {name: histogram[value] for name, value in _SUMMARY_SCHEDULE_RECORDS}

    def support_early_checkpoint_removal(self) -> bool:
        """
        Early checkpoint removal currently not supported for DyHPO
        """
        return False
