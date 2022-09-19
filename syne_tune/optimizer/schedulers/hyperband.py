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
import copy
import logging
import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.hyperband_cost_promotion import (
    CostPromotionRungSystem,
)
from syne_tune.optimizer.schedulers.hyperband_pasha import PASHARungSystem
from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem
from syne_tune.optimizer.schedulers.hyperband_rush import (
    RUSHPromotionRungSystem,
    RUSHStoppingRungSystem,
)
from syne_tune.optimizer.schedulers.hyperband_stopping import StoppingRungSystem
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    Integer,
    Boolean,
    Categorical,
    filter_by_key,
    String,
    Dictionary,
    Float,
)
from syne_tune.optimizer.schedulers.searchers.bracket_distribution import (
    DefaultHyperbandBracketDistribution,
)

__all__ = [
    "HyperbandScheduler",
    "HyperbandBracketManager",
    "hyperband_rung_levels",
]

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    "resource_attr",
    "grace_period",
    "reduction_factor",
    "brackets",
    "type",
    "searcher_data",
    "cost_attr",
    "register_pending_myopic",
    "do_snapshots",
    "rung_system_per_bracket",
    "rung_levels",
    "rung_system_kwargs",
}

_DEFAULT_OPTIONS = {
    "resource_attr": "epoch",
    "resume": False,
    "grace_period": 1,
    "reduction_factor": 3,
    "brackets": 1,
    "type": "stopping",
    "searcher_data": "rungs",
    "register_pending_myopic": False,
    "do_snapshots": False,
    "rung_system_per_bracket": False,
    "rung_system_kwargs": {
        "ranking_criterion": "soft_ranking",
        "epsilon": 1.0,
        "epsilon_scaling": 1.0,
    },
}

_CONSTRAINTS = {
    "resource_attr": String(),
    "resume": Boolean(),
    "grace_period": Integer(1, None),
    "reduction_factor": Float(2, None),
    "brackets": Integer(1, None),
    "type": Categorical(
        (
            "stopping",
            "promotion",
            "cost_promotion",
            "pasha",
            "rush_promotion",
            "rush_stopping",
        )
    ),
    "searcher_data": Categorical(("rungs", "all", "rungs_and_last")),
    "cost_attr": String(),
    "register_pending_myopic": Boolean(),
    "do_snapshots": Boolean(),
    "rung_system_per_bracket": Boolean(),
    "rung_system_kwargs": Dictionary(),
}


def is_continue_decision(trial_decision: str) -> bool:
    return trial_decision == SchedulerDecision.CONTINUE


@dataclass
class TrialInformation:
    """
    The scheduler maintains information about all trials it has been dealing
    with so far. `trial_decision` is the current status of the trial.
    `keep_case` is relevant only if `searcher_data == 'rungs_and_last'`.
    `largest_update_resource` is the largest resource level for which the
    searcher was updated, or None.
    `reported_result` caontains the last recent reported result, or None
    (task was started, but did not report anything yet). Only contains
    attributes `self.metric` and `self._resource_attr`.
    """

    config: dict
    time_stamp: float
    bracket: int
    keep_case: bool
    trial_decision: str
    reported_result: Optional[dict] = None
    largest_update_resource: Optional[int] = None

    def restart(self, time_stamp: float):
        self.time_stamp = time_stamp
        self.reported_result = None
        self.keep_case = False
        self.trial_decision = SchedulerDecision.CONTINUE


class HyperbandScheduler(FIFOScheduler):
    r"""Implements different variants of asynchronous Hyperband

    See 'type' for the different variants. One implementation detail is when
    using multiple brackets, task allocation to bracket is done randomly,
    based on a distribution inspired by the synchronous Hyperband case.

    For definitions of concepts (bracket, rung, milestone), see

        Li, Jamieson, Rostamizadeh, Gonina, Hardt, Recht, Talwalkar (2018)
        A System for Massively Parallel Hyperparameter Tuning
        https://arxiv.org/abs/1810.05934

    or

        Tiao, Klein, Lienart, Archambeau, Seeger (2020)
        Model-based Asynchronous Hyperparameter and Neural Architecture Search
        https://arxiv.org/abs/2003.10865

    Note: This scheduler requires both `metric` and `resource_attr` to be
    returned by the reporter. Here, resource values must be positive int. If
    resource_attr == 'epoch', this should be the number of epochs done,
    starting from 1 (not the epoch number, starting from 0).

    Rung levels and promotion quantiles:

    Rung levels are values of the resource attribute at which stop/go decisions
    are made for jobs, comparing their metric against others at the same level.
    These rung levels (positive, strictly increasing) can be specified via
    `rung_levels`, the largest must be `<= max_t`.
    If `rung_levels` is not given, rung levels are specified by `grace_period`
    and `reduction_factor`:

        [round(grace_period * (reduction_factor ** j))], j = 0, 1, ...

    This is the default choice for successive halving (Hyperband).
    Note: If `rung_levels` is given, then `grace_period`, `reduction_factor`
    are ignored. If they are given, a warning is logged.

    The rung levels determine the quantiles to be used in the stop/go
    decisions. If rung levels are r_0, r_1, ..., define

        q_j = r_j / r_{j+1}

    q_j is the promotion quantile at rung level r_j. On average, a fraction
    of q_j jobs can continue, the remaining ones are stopped (or paused).
    In the default successive halving case:

        q_j = 1 / reduction_factor    for all j

    Cost-aware schedulers or searchers:

    Some schedulers (e.g., type == 'cost_promotion') or searchers may depend
    on cost values (with key `cost_attr`) reported alongside the target metric.
    For promotion-based scheduling, a trial may pause and resume several times.
    The cost received in `on_trial_result` only counts the cost since the last
    resume. We maintain the sum of such costs in `_cost_offset`, and append
    a new entry to `result` in `on_trial_result` with the total cost.
    If the evaluation function does not implement checkpointing, once a trial
    is resumed, it has to start from scratch. We detect this in
    `on_trial_result` and reset the cost offset to 0 (if the trial runs from
    scratch, the cost reported needs no offset added).
    NOTE: This process requires `cost_attr` to be set!

    Pending evaluations:

    The searcher is notified. by `searcher.register_pending` calls, of
    (trial, resource) pairs for which evaluations are running, and a result
    is expected in the future. These pending evaluations can be used by the
    searcher in order to direct sampling elsewhere.
    The choice of pending evaluations depends on `searcher_data`. If equal
    to `'rungs'`, pending evaluations sit only at rung levels, because
    observations are only used there. In the other cases, pending evaluations
    sit at all resource levels for which observations are obtained. For
    example, if a trial is at rung level `r` and continues towards the next
    rung level `r_next`, if `searcher_data == 'rungs'`,
    `searcher.register_pending` is called for `r_next` only, while for other
    `searcher_data` values, pending evaluations are registered for
    `r + 1, r + 2, ..., r_next`.
    However, if in this case, `register_pending_myopic` is True, we instead
    call `searcher.register_pending` for `r + 1` when each observation is
    obtained (not just at a rung level). This leads to less pending
    evaluations at any one time. On the other hand, when a trial is continued
    at a rung level, we already know it will emit observations up to the next
    rung level, so it seems more "correct" to register all these pending
    evaluations in one go.

    Parameters
    ----------
    config_space: dict
        Configuration space for trial evaluation function
    searcher : str or BaseSearcher
        Searcher (get_config decisions). If str, this is passed to
        searcher_factory along with search_options.
    search_options : dict
        If searcher is str, these arguments are passed to searcher_factory.
    checkpoint : str
        If filename given here, a checkpoint of scheduler (and searcher) state
        is written to file every time a job finishes.
        Note: May not be fully supported by all searchers.
    resume : bool
        If True, scheduler state is loaded from checkpoint, and experiment
        starts from there.
        Note: May not be fully supported by all searchers.
    metric : str
        Name of metric to optimize, key in result's obtained via
        `on_trial_result`
    mode : str
        See :class:`TrialScheduler`
    resource_attr : str
        Name of resource attribute in result's obtained via `on_trial_result`.
        Note: The type of resource must be int.
    points_to_evaluate : List[dict] or None
        See :class:`FIFOScheduler`
    max_t : int
        See :class:`FIFOScheduler`. This is mandatory here. If not given, we
        try to infer it.
    grace_period : int
        Minimum resource to be used for a job. Ignored if `rung_levels` is
        given.
    reduction_factor : float (>= 2)
        Parameter to determine rung levels in successive halving (Hyperband).
        Ignored if `rung_levels` is given.
    rung_levels: list of int
        If given, prescribes the set of rung levels to be used. Must contain
        positive integers, strictly increasing. This information overrides
        `grace_period` and `reduction_factor`, but not `max_t`.
        Note that the stop/promote rule in the successive halving scheduler is
        set based on the ratio of successive rung levels.
    brackets : int
        Number of brackets to be used in Hyperband. Each bracket has a different
        grace period, all share max_t and reduction_factor.
        If brackets == 1, we run successive halving.
    extra_searcher_info : bool
        If True, information about the current state of the searcher returned
        in `on_trial_result`. This info includes in particular the current
        hyperparameters of the surrogate model of the searcher, as well as the
        dataset size.
    type : str
        Type of Hyperband scheduler:
            stopping:
                A config eval is executed by a single task. The task is stopped
                at a milestone if its metric is worse than a fraction of those
                who reached the milestone earlier, otherwise it continues.
                As implemented in Ray/Tune:
                https://ray.readthedocs.io/en/latest/tune-schedulers.html#asynchronous-hyperband
                See :class:`StoppingRungSystem`.
            promotion:
                A config eval may be associated with multiple tasks over its
                lifetime. It is never terminated, but may be paused. Whenever a
                task becomes available, it may promote a config to the next
                milestone, if better than a fraction of others who reached the
                milestone. If no config can be promoted, a new one is chosen.
                This variant may benefit from pause&resume, which is not directly
                supported here. As proposed in this paper (termed ASHA):
                https://arxiv.org/abs/1810.05934
                See :class:`PromotionRungSystem`.
            cost_promotion:
                This is a cost-aware variant of 'promotion', see
                :class:`CostPromotionRungSystem` for details. In this case,
                costs must be reported under the name
                `rung_system_kwargs['cost_attr']` in results.
            pasha:
                Similar to promotion type Hyperband, but it progressively
                expands the available resources until the ranking
                of configurations stabilizes.
            rush_stopping:
                A variation of the stopping scheduler which requires passing rung_system_kwargs
                (see num_threshold_candidates) and points_to_evaluate. The first num_threshold_candidates of
                points_to_evaluate will enforce stricter rules on which task is continued.
                See :class:`RUSHScheduler`.
            rush_promotion:
                Same as rush_stopping but for promotion.
    cost_attr : str
        Required if the scheduler itself uses a cost metric (i.e.,
        `type='cost_promotion'`), or if the searcher uses a cost metric.
        See also header comment.
    searcher_data : str
        Relevant only if a model-based searcher is used.
        Example: For NN tuning and `resource_attr == epoch', we receive a
        result for each epoch, but not all epoch values are also rung levels.
        searcher_data determines which of these results are passed to the
        searcher. As a rule, the more data the searcher receives, the better
        its fit, but also the more expensive get_config may become. Choices:
        - 'rungs' (default): Only results at rung levels. Cheapest
        - 'all': All results. Most expensive
        - 'rungs_and_last': Results at rung levels, plus the most recent
            result. This means that in between rung levels, only the most
            recent result is used by the searcher. This is in between
        Note: For a Gaussian additive learning curve surrogate model, this
        has to be set to 'all'.
    register_pending_myopic : bool
        See above. Used only if `searcher_data != 'rungs'`.
    rung_system_per_bracket : bool
        This concerns Hyperband with `brackets > 1`. When starting a job for a
        new config, it is assigned a randomly sampled bracket. The larger the
        bracket, the larger the grace period for the config. If
        `rung_system_per_bracket = True`, we maintain separate rung level
        systems for each bracket, so that configs only compete with others
        started in the same bracket.
        If `rung_system_per_bracket = False`, we use a single rung level system,
        so that all configs compete with each other. In this case, the bracket
        of a config only determines the initial grace period, i.e. the first
        milestone at which it starts competing with others. This is the
        default.
        The concept of brackets in Hyperband is meant to hedge against overly
        aggressive filtering in successive halving, based on low fidelity
        criteria. In practice, successive halving (i.e., `brackets = 1`) often
        works best in the asynchronous case (as implemented here). If
        `brackets > 1`, the hedging is stronger if `rung_system_per_bracket`
        is True.
    do_snapshots : bool
        Support snapshots? If True, a snapshot of all running tasks and rung
        levels is returned by _promote_config. This snapshot is passed to the
        searcher in get_config.
        Note: Currently, only the stopping variant supports snapshots.
    max_resource_attr : str
        Optional. Relevant only for type 'promotion'. Key name in config for
        fixed attribute containing the maximum resource. The training
        evaluation function runs a loop over 1, ..., config[max_resource_attr],
        or starts from a resource > 1 if a checkpoint can be loaded.
        Whenever a trial is started or resumed here, this value in the config
        is set to the next rung level this trial will reach. As it will pause
        there in any case, this precludes the training code to continue until a
        stop signal is received.
        If given, `max_resource_attr` is also used in the mechanism to infer
        `max_t` (if not given).
    rung_system_kwargs : dict
        Arguments passed to the rung system:
            ranking_criterion : str
                Used if `type == 'pasha'`. Specifies what strategy to use
                for deciding if the ranking is stable and if to increase the resource.
                Available options are soft_ranking, soft_ranking_std,
                soft_ranking_median_dst and soft_ranking_mean_dst. The simplest
                soft_ranking accepts a manually specified value of epsilon and
                groups configurations with similar performance within the given range
                of objective values. The other strategies calculate the value of epsilon
                automatically, with the option to rescale the it using epsilon_scaling.
            epsilon : float
                Used if `type == 'pasha'`. Parameter for soft ranking in PASHA
                to say which configurations should be group together based on the
                similarity of their performance.
            epsilon_scaling : float
                Used if `type == 'pasha'`. When epsilon for soft ranking in
                PASHA is calculated automatically, it is possible to rescale it
                using epsilon_scaling.
            num_threshold_candidates : int
                Used if `type in ['rush_promotion', 'rush_stopping']`. The first num_threshold_candidates in
                points_to_evaluate enforce stricter requirements to the continuation of training tasks.
                See :class:`RUSHScheduler`.

    See Also
    --------
    HyperbandBracketManager
    """

    def __init__(self, config_space, **kwargs):
        # Before we can call the superclass constructor, we need to set a few
        # members (see also `_extend_search_options`).
        # To do this properly, we first check values and impute defaults for
        # `kwargs`.
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
        scheduler_type = kwargs["type"]
        self.scheduler_type = scheduler_type
        self._resource_attr = kwargs["resource_attr"]
        self._rung_system_kwargs = kwargs["rung_system_kwargs"]
        self._cost_attr = kwargs.get("cost_attr")
        self._num_brackets = kwargs["brackets"]
        assert not (
            scheduler_type == "cost_promotion" and self._cost_attr is None
        ), "cost_attr must be given if type='cost_promotion'"
        # Default: May be modified by searcher (via `configure_scheduler`)
        self.bracket_distribution = DefaultHyperbandBracketDistribution()
        # Superclass constructor
        resume = kwargs["resume"]
        kwargs["resume"] = False  # Cannot be done in superclass
        super().__init__(config_space, **filter_by_key(kwargs, _ARGUMENT_KEYS))
        assert self.max_t is not None, (
            "Either max_t must be specified, or it has to be specified as "
            + "config_space['epochs'], config_space['max_t'], "
            + "config_space['max_epochs']"
        )

        # If rung_levels is given, grace_period and reduction_factor are ignored
        rung_levels = kwargs.get("rung_levels")
        if rung_levels is not None:
            assert isinstance(rung_levels, list)
            if ("grace_period" in kwargs) or ("reduction_factor" in kwargs):
                logger.warning(
                    "Since rung_levels is given, the values grace_period = "
                    f"{kwargs.get('grace_period')} and reduction_factor = "
                    f"{kwargs.get('reduction_factor')} are ignored!"
                )
        rung_levels = hyperband_rung_levels(
            rung_levels,
            grace_period=kwargs["grace_period"],
            reduction_factor=kwargs["reduction_factor"],
            max_t=self.max_t,
        )
        do_snapshots = kwargs["do_snapshots"]
        assert (not do_snapshots) or (
            scheduler_type == "stopping"
        ), "Snapshots are supported only for type = 'stopping'"
        rung_system_per_bracket = kwargs["rung_system_per_bracket"]

        self.terminator = HyperbandBracketManager(
            scheduler_type,
            self._resource_attr,
            self.metric,
            self.mode,
            self.max_t,
            rung_levels,
            self._num_brackets,
            rung_system_per_bracket,
            cost_attr=self._total_cost_attr(),
            random_seed=self.random_seed_generator(),
            rung_system_kwargs=self._rung_system_kwargs,
            scheduler=self,
        )
        self.do_snapshots = do_snapshots
        self.searcher_data = kwargs["searcher_data"]
        self._register_pending_myopic = kwargs["register_pending_myopic"]
        # _active_trials:
        # Maintains information for all tasks (running, paused, or stopped).
        # Maps trial_id to `TrialInformation`.
        self._active_trials = dict()
        # _cost_offset:
        # Is used for promotion-based (pause/resume) scheduling if the eval
        # function reports cost values. For a trial which has been paused at
        # least once, this records the sum of costs for reaching its last
        # recent milestone.
        self._cost_offset = dict()
        if resume:
            checkpoint = kwargs.get("checkpoint")
            assert checkpoint is not None, "Need checkpoint to be set if resume = True"
            if os.path.isfile(checkpoint):
                raise NotImplementedError()
                # TODO! Need load
                # self.load_state_dict(load(checkpoint))
            else:
                msg = f"checkpoint path {checkpoint} is not available for resume."
                logger.exception(msg)
                raise FileExistsError(msg)

    def does_pause_resume(self) -> bool:
        """
        :return: Is this variant doing pause and resume scheduling, in the
            sense that trials can be paused and resumed later?
        """
        return self.scheduler_type != "stopping"

    @property
    def rung_levels(self) -> List[int]:
        return self.terminator.rung_levels

    def _initialize_searcher(self):
        if not self._searcher_initialized:
            super()._initialize_searcher()
            self.bracket_distribution.configure(self)

    def _extend_search_options(self, search_options: dict) -> dict:
        # Note: Needs self.scheduler_type to be set
        scheduler = "hyperband_{}".format(self.scheduler_type)
        result = dict(
            search_options, scheduler=scheduler, resource_attr=self._resource_attr
        )
        # Cost attribute: For promotion-based, cost needs to be accumulated
        # for each trial
        cost_attr = self._total_cost_attr()
        if cost_attr is not None:
            result["cost_attr"] = cost_attr
        if "hypertune_distribution_num_samples" in result:
            result["hypertune_distribution_num_brackets"] = self._num_brackets
        return result

    def _total_cost_attr(self) -> Optional[str]:
        if self._cost_attr is None:
            return None
        elif self.does_pause_resume():
            return "total_" + self._cost_attr
        else:
            return self._cost_attr

    def _on_config_suggest(self, config: dict, trial_id: str, **kwargs) -> dict:
        """
        `kwargs` being used here:
        - elapsed_time: Time from start of experiment, set in
            `FIFOScheduler._suggest`
        - bracket: Bracket in which new trial is started, set in
            `HyperbandScheduler._promote_trial`
        - milestone: First milestone the new trial will reach, set in
            `HyperbandScheduler._promote_trial`
        """
        assert trial_id not in self._active_trials, f"Trial {trial_id} already exists"
        # See `FIFOScheduler._on_config_suggest` for why we register the task
        # and pending evaluation here, and not later in `on_task_add`.
        debug_log = self.searcher.debug_log
        # Register new task
        first_milestone = self.terminator.on_task_add(
            trial_id, bracket=kwargs["bracket"], new_config=True
        )[-1]
        if debug_log is not None:
            logger.info(
                f"trial_id {trial_id} starts (first milestone = " f"{first_milestone})"
            )
        # Register pending evaluation with searcher
        if self.searcher_data == "rungs":
            pending_resources = [first_milestone]
        elif self._register_pending_myopic:
            pending_resources = [1]
        else:
            pending_resources = list(range(1, first_milestone + 1))
        for resource in pending_resources:
            self.searcher.register_pending(
                trial_id=trial_id, config=config, milestone=resource
            )
        # Extra fields in `config`
        if debug_log is not None:
            # For log outputs:
            config["trial_id"] = trial_id
        if self.does_pause_resume() and self.max_resource_attr is not None:
            # The new trial should only run until the next milestone.
            # This needs its config to be modified accordingly.
            config[self.max_resource_attr] = kwargs["milestone"]

        self._active_trials[trial_id] = TrialInformation(
            config=copy.copy(config),
            time_stamp=kwargs["elapsed_time"],
            bracket=kwargs["bracket"],
            keep_case=False,
            trial_decision=SchedulerDecision.CONTINUE,
            largest_update_resource=None,
        )

        return config

    # Snapshot (in extra_kwargs['snapshot']):
    # - max_resource
    # - reduction_factor
    # - tasks: Info about running tasks in bracket bracket_id (or, if
    #   brackets share the same rung level system, all running tasks):
    #   dict(task_id) -> dict:
    #   - config: config as dict
    #   - time: Time when task was started, or when last recent result was
    #     reported
    #   - level: Level of last recent result report, or 0 if no reports yet
    # - rungs: Metric values at rung levels in bracket bracket_id:
    #   List of (rung_level, metric_dict), where metric_dict has entries
    #   task_id: metric_value. Note that entries are sorted in decreasing order
    #   w.r.t. rung_level.
    def _promote_trial(self) -> (Optional[str], Optional[dict]):
        trial_id, extra_kwargs = self.terminator.on_task_schedule()
        if trial_id is None:
            # No trial to be promoted
            if self.do_snapshots:
                # Append snapshot
                bracket_id = extra_kwargs["bracket"]
                extra_kwargs["snapshot"] = {
                    "tasks": self._snapshot_tasks(bracket_id),
                    "rungs": self.terminator.snapshot_rungs(bracket_id),
                    "max_resource": self.max_t,
                }
        else:
            # At this point, we can assume the trial will be resumed
            extra_kwargs["new_config"] = False
            self.terminator.on_task_add(trial_id, **extra_kwargs)
            # Update information (note that 'time_stamp' is not exactly
            # correct, since the task may get started a little later)
            assert (
                trial_id in self._active_trials
            ), f"Paused trial {trial_id} must be in _active_trials"
            record = self._active_trials[trial_id]
            assert not is_continue_decision(
                record.trial_decision
            ), f"Paused trial {trial_id} marked as running in _active_trials"
            record.restart(time_stamp=self._elapsed_time())
            # Register pending evaluation(s) with searcher
            next_milestone = extra_kwargs["milestone"]
            resume_from = extra_kwargs["resume_from"]
            if self.searcher_data == "rungs":
                pending_resources = [next_milestone]
            elif self._register_pending_myopic:
                pending_resources = [resume_from + 1]
            else:
                pending_resources = list(range(resume_from + 1, next_milestone + 1))
            for resource in pending_resources:
                self.searcher.register_pending(trial_id=trial_id, milestone=resource)
            if self.searcher.debug_log is not None:
                logger.info(
                    f"trial_id {trial_id}: Promotion from "
                    f"{resume_from} to {next_milestone}"
                )
            # In the case of a promoted trial, extra_kwargs plays a different
            # role
            if self.does_pause_resume() and self.max_resource_attr is not None:
                # The promoted trial should only run until the next milestone.
                # This needs its config to be modified accordingly
                extra_kwargs = record.config.copy()
                extra_kwargs[self.max_resource_attr] = next_milestone
            else:
                extra_kwargs = None
        return trial_id, extra_kwargs

    def _snapshot_tasks(self, bracket_id):
        # If all brackets share a single rung level system, then all
        # running jobs have to be taken into account, otherwise only
        # those jobs running in the same bracket
        all_running = not self.terminator._rung_system_per_bracket
        tasks = dict()
        for k, v in self._active_trials.items():
            if is_continue_decision(v.trial_decision) and (
                all_running or v.bracket == bracket_id
            ):
                reported_result = v.reported_result
                level = (
                    0
                    if reported_result is None
                    else reported_result[self._resource_attr]
                )
                # It is possible to have tasks in _active_trials which have
                # reached self.max_t. These must not end up in the snapshot
                if level < self.max_t:
                    tasks[k] = {
                        "config": v.config,
                        "time": v.time_stamp,
                        "level": level,
                    }
        return tasks

    def _cleanup_trial(self, trial_id: str, trial_decision: str):
        """
        Called for trials which are stopped or paused. The trial is still kept
        in the records.

        :param trial_id:
        """
        self.terminator.on_task_remove(trial_id)
        if trial_id in self._active_trials:
            # We do not remove stopped trials
            self._active_trials[trial_id].trial_decision = trial_decision

    def on_trial_error(self, trial: Trial):
        super().on_trial_error(trial)
        self._cleanup_trial(str(trial.trial_id), trial_decision=SchedulerDecision.STOP)

    def _update_searcher_internal(self, trial_id: str, config: dict, result: dict):
        if self.searcher_data == "rungs_and_last":
            # Remove last recently added result for this task. This is not
            # done if it fell on a rung level (i.e., `keep_case` is True)
            record = self._active_trials[trial_id]
            rem_result = record.reported_result
            if (rem_result is not None) and (not record.keep_case):
                self.searcher.remove_case(trial_id, **rem_result)

    def _update_searcher(
        self, trial_id: str, config: dict, result: dict, task_info: dict
    ):
        """
        Updates searcher with `result` (depending on `searcher_data`), and
        registers pending config with searcher.

        :param trial_id:
        :param config:
        :param result: Record obtained from `on_trial_result`
        :param task_info: Info from `self.terminator.on_task_report`
        :return: Should searcher be updated?
        """
        task_continues = task_info["task_continues"]
        milestone_reached = task_info["milestone_reached"]
        next_milestone = task_info.get("next_milestone")
        do_update = False
        pending_resources = []
        if self.searcher_data == "rungs":
            resource = result[self._resource_attr]
            if resource in self.rung_levels or resource == self.max_t:
                # Update searcher with intermediate result
                # Note: This condition is weaker than `milestone_reached` if
                # more than one bracket is used
                do_update = True
                if task_continues and milestone_reached and next_milestone is not None:
                    pending_resources = [next_milestone]
        elif not task_info.get("ignore_data", False):
            # All results are reported to the searcher, except if
            # task_info['ignore_data'] is True. The latter happens only for
            # tasks running promoted configs. In this case, we may receive
            # reports before the first milestone is reached, which should not
            # be passed to the searcher (they'd duplicate earlier
            # datapoints).
            # See also header comment of PromotionRungSystem.
            do_update = True
            if task_continues:
                resource = int(result[self._resource_attr])
                if self._register_pending_myopic or next_milestone is None:
                    pending_resources = [resource + 1]
                elif milestone_reached:
                    # Register pending evaluations for all resources up to
                    # `next_milestone`
                    pending_resources = list(range(resource + 1, next_milestone + 1))
        # Update searcher
        if do_update:
            self._update_searcher_internal(trial_id, config, result)
        # Register pending evaluations
        for resource in pending_resources:
            self.searcher.register_pending(
                trial_id=trial_id, config=config, milestone=resource
            )
        return do_update

    def _check_result(self, result: dict):
        super()._check_result(result)
        self._check_key_of_result(result, self._resource_attr)
        if self.scheduler_type == "cost_promotion":
            self._check_key_of_result(result, self._cost_attr)
        resource = result[self._resource_attr]
        assert 1 <= resource == round(resource), (
            "Your training evaluation function needs to report positive "
            + f"integer values for key {self._resource_attr}. Obtained "
            + f"value {resource}, which is not permitted"
        )

    def on_trial_result(self, trial: Trial, result: dict) -> str:
        self._check_result(result)
        trial_id = str(trial.trial_id)
        debug_log = self.searcher.debug_log
        trial_decision = SchedulerDecision.CONTINUE
        if len(result) == 0:
            # An empty dict should just be skipped
            if debug_log is not None:
                logger.info(
                    f"trial_id {trial_id}: Skipping empty dict received "
                    "from reporter"
                )
        else:
            # Time since start of experiment
            time_since_start = self._elapsed_time()
            do_update = False
            config = self._preprocess_config(trial.config)
            cost_and_promotion = (
                self._cost_attr is not None
                and self._cost_attr in result
                and self.does_pause_resume()
            )
            if cost_and_promotion:
                # Trial may have paused/resumed before, so need to add cost
                # offset from these
                cost_offset = self._cost_offset.get(trial_id, 0)
                result[self._total_cost_attr()] = result[self._cost_attr] + cost_offset
            # We may receive a report from a trial which has been stopped or
            # paused before. In such a case, we override trial_decision to be
            # STOP or PAUSE as before, so the report is not taken into account
            # by the scheduler. The report is sent to the searcher, but with
            # update=False. This means that the report is registered, but cannot
            # influence any decisions.
            record = self._active_trials[trial_id]
            trial_decision = record.trial_decision
            if trial_decision != SchedulerDecision.CONTINUE:
                logger.warning(
                    f"trial_id {trial_id}: {trial_decision}, but receives "
                    f"another report {result}\nThis report is ignored"
                )
            else:
                task_info = self.terminator.on_task_report(trial_id, result)
                task_continues = task_info["task_continues"]
                milestone_reached = task_info["milestone_reached"]
                if cost_and_promotion:
                    if milestone_reached:
                        # Trial reached milestone and will pause there: Update
                        # cost offset
                        if self._cost_attr is not None:
                            self._cost_offset[trial_id] = result[
                                self._total_cost_attr()
                            ]
                    elif task_info.get("ignore_data", False):
                        # For a resumed trial, the report is for resource <=
                        # resume_from, where resume_from < milestone. This
                        # happens if checkpointing is not implemented and a
                        # resumed trial has to start from scratch, publishing
                        # results all the way up to resume_from. In this case,
                        # we can erase the `_cost_offset` entry, since the
                        # instantaneous cost reported by the trial does not
                        # have any offset.
                        if self._cost_offset[trial_id] > 0:
                            logger.info(
                                f"trial_id {trial_id}: Resumed trial seems to have been "
                                + "started from scratch (no checkpointing?), so we erase "
                                + "the cost offset."
                            )
                        self._cost_offset[trial_id] = 0

                # Update searcher and register pending
                do_update = self._update_searcher(trial_id, config, result, task_info)
                # Change snapshot entry for task
                # Note: This must not be done above, because what _update_searcher
                # is doing, depends on the entry *before* its update here.
                # Note: result may contain all sorts of extra info.
                # All we need to maintain in the snapshot are metric and
                # resource level.
                # 'keep_case' entry (only used if searcher_data ==
                # 'rungs_and_last'): The result is kept in the dataset iff
                # milestone_reached == True (i.e., we are at a rung level).
                # Otherwise, it is removed once _update_searcher is called for
                # the next recent result.
                resource = int(result[self._resource_attr])
                record.time_stamp = time_since_start
                record.reported_result = {
                    self.metric: result[self.metric],
                    self._resource_attr: resource,
                }
                record.keep_case = milestone_reached
                if do_update:
                    largest_update_resource = record.largest_update_resource
                    if largest_update_resource is None:
                        largest_update_resource = resource - 1
                    assert largest_update_resource <= resource, (
                        f"Internal error (trial_id {trial_id}): "
                        + f"on_trial_result called with resource = {resource}, "
                        + f"but largest_update_resource = {largest_update_resource}"
                    )
                    if resource == largest_update_resource:
                        do_update = False  # Do not update again
                    else:
                        record.largest_update_resource = resource
                if not task_continues:
                    if (not self.does_pause_resume()) or resource >= self.max_t:
                        trial_decision = SchedulerDecision.STOP
                        act_str = "Terminating"
                    else:
                        trial_decision = SchedulerDecision.PAUSE
                        act_str = "Pausing"
                    self._cleanup_trial(trial_id, trial_decision=trial_decision)
                if debug_log is not None:
                    if not task_continues:
                        logger.info(
                            f"trial_id {trial_id}: {act_str} evaluation "
                            f"at {resource}"
                        )
                    elif milestone_reached:
                        msg = f"trial_id {trial_id}: Reaches {resource}, continues"
                        next_milestone = task_info.get("next_milestone")
                        if next_milestone is not None:
                            msg += f" to {next_milestone}"
                        logger.info(msg)
            self.searcher.on_trial_result(
                trial_id, config, result=result, update=do_update
            )
        # Extra info in debug mode
        log_msg = f"trial_id {trial_id} (metric = {result[self.metric]:.3f}"
        for k, is_float in ((self._resource_attr, False), ("elapsed_time", True)):
            if k in result:
                if is_float:
                    log_msg += f", {k} = {result[k]:.2f}"
                else:
                    log_msg += f", {k} = {result[k]}"
        log_msg += f"): decision = {trial_decision}"
        logger.debug(log_msg)
        return trial_decision

    def on_trial_remove(self, trial: Trial):
        self._cleanup_trial(str(trial.trial_id), trial_decision=SchedulerDecision.PAUSE)

    def on_trial_complete(self, trial: Trial, result: dict):
        # Check whether searcher was already updated based on `result`
        trial_id = str(trial.trial_id)
        largest_update_resource = self._active_trials[trial_id].largest_update_resource
        if largest_update_resource is not None:
            resource = int(result[self._resource_attr])
            if resource > largest_update_resource:
                super().on_trial_complete(trial, result)
        # Remove pending evaluations, in case there are still some
        self.searcher.cleanup_pending(trial_id)
        self._cleanup_trial(trial_id, trial_decision=SchedulerDecision.STOP)


def _is_positive_int(x):
    return int(x) == x and x >= 1


def hyperband_rung_levels(rung_levels, grace_period, reduction_factor, max_t):
    if rung_levels is not None:
        assert (
            isinstance(rung_levels, list) and len(rung_levels) > 1
        ), "rung_levels must be list of size >= 2"
        assert all(
            _is_positive_int(x) for x in rung_levels
        ), "rung_levels must be list of positive integers"
        rung_levels = [int(x) for x in rung_levels]
        assert all(
            x < y for x, y in zip(rung_levels, rung_levels[1:])
        ), "rung_levels must be strictly increasing sequence"
        assert (
            rung_levels[-1] <= max_t
        ), f"Last entry of rung_levels ({rung_levels[-1]}) must be <= max_t ({max_t})"
    else:
        # Rung levels given by grace_period, reduction_factor, max_t
        assert _is_positive_int(grace_period)
        assert reduction_factor >= 2
        assert _is_positive_int(max_t)
        assert (
            max_t > grace_period
        ), f"max_t ({max_t}) must be greater than grace_period ({grace_period})"
        rf = reduction_factor
        min_t = grace_period
        max_rungs = int(np.log(max_t / min_t) / np.log(rf) + 1)
        rung_levels = [int(round(min_t * np.power(rf, k))) for k in range(max_rungs)]
        assert rung_levels[-1] <= max_t  # Sanity check
        assert len(rung_levels) >= 2, (
            f"grace_period = {grace_period}, reduction_factor = "
            + f"{reduction_factor}, max_t = {max_t} leads to single rung level only"
        )
    return rung_levels


class HyperbandBracketManager:
    """Hyperband Manager

    Maintains rung level systems for range of brackets. Differences depending
    on `scheduler_type` ('stopping', 'promotion') manifest themselves mostly
    at the level of the rung level system itself.

    For `scheduler_type` == 'stopping', see :class:`StoppingRungSystem`.
    For `scheduler_type` == 'promotion', see :class:`PromotionRungSystem`.

    Args:
        scheduler_type : str
            See :class:`HyperbandScheduler`.
        resource_attr : str
            See :class:`HyperbandScheduler`.
        metric : str
            See :class:`HyperbandScheduler`.
        mode : str
            See :class:`HyperbandScheduler`.
        max_t : int
            See :class:`HyperbandScheduler`.
        rung_levels : list[int]
            See :class:`HyperbandScheduler`. If `rung_levels` is not given
            there, the default rung levels based on `grace_period` and
            `reduction_factor` are used.
        brackets : int
            See :class:`HyperbandScheduler`.
        rung_system_per_bracket : bool
            See :class:`HyperbandScheduler`.
        cost_attr : str
            Overrides entry in `rung_system_kwargs`
        random_seed : int
            Random seed for bracket sampling
        rung_system_kwargs : dict
            dictionary of arguments passed to the rung system
        scheduler : HyperbandScheduler
            The scheduler is needed in order to sample a bracket
    """

    def __init__(
        self,
        scheduler_type,
        resource_attr,
        metric,
        mode,
        max_t,
        rung_levels,
        brackets,
        rung_system_per_bracket,
        cost_attr,
        random_seed,
        rung_system_kwargs,
        scheduler,
    ):
        self._scheduler_type = scheduler_type
        self._resource_attr = resource_attr
        self._max_t = max_t
        self.rung_levels = copy.copy(rung_levels)
        self._rung_system_per_bracket = rung_system_per_bracket
        self._scheduler = scheduler
        # Maps trial_id -> bracket_id
        self._task_info = dict()
        max_num_brackets = len(rung_levels)
        self.num_brackets = min(brackets, max_num_brackets)
        num_systems = self.num_brackets if rung_system_per_bracket else 1
        rung_levels_plus_maxt = rung_levels[1:] + [max_t]
        # Promotion quantiles: q_j = r_j / r_{j+1}
        promote_quantiles = [x / y for x, y in zip(rung_levels, rung_levels_plus_maxt)]
        kwargs = dict(metric=metric, mode=mode, resource_attr=resource_attr)
        if scheduler_type == "stopping":
            rs_type = StoppingRungSystem
        elif scheduler_type == "pasha":
            kwargs["max_t"] = max_t
            kwargs["ranking_criterion"] = rung_system_kwargs["ranking_criterion"]
            kwargs["epsilon"] = rung_system_kwargs["epsilon"]
            kwargs["epsilon_scaling"] = rung_system_kwargs["epsilon_scaling"]
            rs_type = PASHARungSystem
        elif scheduler_type in ["rush_promotion", "rush_stopping"]:
            kwargs["num_threshold_candidates"] = rung_system_kwargs.get(
                "num_threshold_candidates", 0
            )
            if scheduler_type == "rush_stopping":
                rs_type = RUSHStoppingRungSystem
            else:
                kwargs["max_t"] = max_t
                rs_type = RUSHPromotionRungSystem
        else:
            kwargs["max_t"] = max_t
            if scheduler_type == "promotion":
                rs_type = PromotionRungSystem
            else:
                kwargs["cost_attr"] = cost_attr
                rs_type = CostPromotionRungSystem
        self._rung_systems = [
            rs_type(
                rung_levels=rung_levels[s:],
                promote_quantiles=promote_quantiles[s:],
                **kwargs,
            )
            for s in range(num_systems)
        ]
        self.random_state = np.random.RandomState(random_seed)

    def _get_rung_system_for_bracket_id(self, bracket_id: int):
        if self._rung_system_per_bracket:
            sys_id = bracket_id
            skip_rungs = 0
        else:
            sys_id = 0
            skip_rungs = bracket_id
        return self._rung_systems[sys_id], skip_rungs

    def _get_rung_system(self, trial_id: str):
        bracket_id = self._task_info[trial_id]
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        return rung_sys, bracket_id, skip_rungs

    def on_task_add(self, trial_id: str, **kwargs) -> List[int]:
        """
        Called when new task is started (can be new trial or trial being
        resumed).

        Since the bracket has already been sampled, not much is done here.
        We return the list of milestones for this bracket in reverse
        (decreasing) order. The first entry is max_t, even if it is
        not a milestone in the bracket. This list contains the resource
        levels the task would reach if it ran to max_t without being stopped.

        :param trial_id:
        :param kwargs:
        :return: List of milestones in decreasing order, where max_t is first
        """
        assert "bracket" in kwargs
        bracket_id = kwargs["bracket"]
        self._task_info[trial_id] = bracket_id
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        rung_sys.on_task_add(trial_id, skip_rungs=skip_rungs, **kwargs)
        milestones = rung_sys.get_milestones(skip_rungs)
        if milestones[0] < self._max_t:
            milestones.insert(0, self._max_t)
        return milestones

    def on_task_report(self, trial_id: str, result: dict):
        """
        This method is called by the reporter thread whenever a new metric
        value is received. It returns a dictionary with all the information
        needed for making decisions (e.g., stop / continue task, update
        model, etc)
        - task_continues: Should task continue or stop/pause?
        - milestone_reached: True if rung level (or max_t) is hit
        - next_milestone: If hit rung level < max_t, this is the subsequent
          rung level (otherwise: None)
        - bracket_id: Bracket in which the task is running

        :param trial_id:
        :param result:
        :return: See above
        """
        rung_sys, bracket_id, skip_rungs = self._get_rung_system(trial_id)
        ret_dict = {
            "bracket_id": bracket_id,
            "task_continues": False,
            "milestone_reached": True,
            "next_milestone": None,
        }
        if self._scheduler_type != "stopping":
            ret_dict["ignore_data"] = False
        if result[self._resource_attr] < self._max_t:
            ret_dict.update(
                rung_sys.on_task_report(trial_id, result, skip_rungs=skip_rungs)
            )
            # Special case: If config just reached the last milestone in
            # the bracket and survived, next_milestone is equal to max_t
            if (
                ret_dict["task_continues"]
                and ret_dict["milestone_reached"]
                and (ret_dict["next_milestone"] is None)
            ):
                ret_dict["next_milestone"] = self._max_t
        return ret_dict

    def on_task_remove(self, trial_id):
        if trial_id in self._task_info:
            rung_sys, _, _ = self._get_rung_system(trial_id)
            rung_sys.on_task_remove(trial_id)
            del self._task_info[trial_id]

    def _sample_bracket(self) -> int:
        distribution = self._scheduler.bracket_distribution()
        return self.random_state.choice(a=distribution.size, p=distribution)

    def on_task_schedule(self) -> (Optional[str], dict):
        """
        Samples bracket for task to be scheduled. Check whether any paused
        trial in that bracket can be promoted. If so, its trial_id is
        returned. We also return extra_kwargs to be used in `_promote_trial`.
        """
        # Sample bracket for task to be scheduled
        bracket_id = self._sample_bracket()
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        extra_kwargs = {"bracket": bracket_id}
        # Check whether config can be promoted
        ret_dict = rung_sys.on_task_schedule()
        trial_id = ret_dict.get("trial_id")
        if trial_id is not None:
            for k in ("milestone", "resume_from"):
                extra_kwargs[k] = ret_dict[k]
        else:
            # First milestone the new config will get to
            extra_kwargs["milestone"] = rung_sys.get_first_milestone(skip_rungs)
        return trial_id, extra_kwargs

    def snapshot_rungs(self, bracket_id):
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        return rung_sys.snapshot_rungs(skip_rungs)
