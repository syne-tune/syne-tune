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
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Callable

import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.multi_fidelity import MultiFidelitySchedulerMixin
from syne_tune.optimizer.schedulers.hyperband_cost_promotion import (
    CostPromotionRungSystem,
)
from syne_tune.optimizer.schedulers.hyperband_pasha import PASHARungSystem
from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem
from syne_tune.optimizer.schedulers.hyperband_rush import (
    RUSHPromotionRungSystem,
    RUSHStoppingRungSystem,
)
from syne_tune.optimizer.schedulers.hyperband_stopping import (
    StoppingRungSystem,
    PausedTrialsResult,
)
from syne_tune.optimizer.schedulers.hyperband_checkpoint_removal import (
    create_callback_for_checkpoint_removal,
)
from syne_tune.optimizer.schedulers.remove_checkpoints import (
    RemoveCheckpointsSchedulerMixin,
)
from syne_tune.optimizer.schedulers.searchers.dyhpo.hyperband_dyhpo import (
    DyHPORungSystem,
    DEFAULT_SH_PROBABILITY,
)
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
from syne_tune.optimizer.schedulers.utils.successive_halving import (
    successive_halving_rung_levels,
)
from syne_tune.tuning_status import TuningStatus
from syne_tune.tuner_callback import TunerCallback

logger = logging.getLogger(__name__)


RUNG_SYSTEMS = {
    "stopping": StoppingRungSystem,
    "promotion": PromotionRungSystem,
    "pasha": PASHARungSystem,
    "rush_promotion": RUSHPromotionRungSystem,
    "rush_stopping": RUSHStoppingRungSystem,
    "cost_promotion": CostPromotionRungSystem,
    "dyhpo": DyHPORungSystem,
}


_ARGUMENT_KEYS = {
    "resource_attr",
    "grace_period",
    "reduction_factor",
    "rung_increment",
    "brackets",
    "type",
    "searcher_data",
    "cost_attr",
    "register_pending_myopic",
    "do_snapshots",
    "rung_system_per_bracket",
    "rung_levels",
    "rung_system_kwargs",
    "early_checkpoint_removal_kwargs",
}

_DEFAULT_OPTIONS = {
    "resource_attr": "epoch",
    "grace_period": 1,
    "brackets": 1,
    "type": "stopping",
    "searcher_data": "rungs",
    "register_pending_myopic": False,
    "do_snapshots": False,
    "rung_system_per_bracket": False,
    "rung_system_kwargs": {
        "num_threshold_candidates": 0,
        "probability_sh": DEFAULT_SH_PROBABILITY,
    },
}

_CONSTRAINTS = {
    "resource_attr": String(),
    "grace_period": Integer(1, None),
    "reduction_factor": Float(2, None),
    "rung_increment": Integer(1, None),
    "brackets": Integer(1, None),
    "type": Categorical(tuple(RUNG_SYSTEMS.keys())),
    "searcher_data": Categorical(("rungs", "all", "rungs_and_last")),
    "cost_attr": String(),
    "register_pending_myopic": Boolean(),
    "do_snapshots": Boolean(),
    "rung_system_per_bracket": Boolean(),
    "rung_system_kwargs": Dictionary(),
    "early_checkpoint_removal_kwargs": Dictionary(),
}


def is_continue_decision(trial_decision: str) -> bool:
    return trial_decision == SchedulerDecision.CONTINUE


@dataclass
class TrialInformation:
    """
    The scheduler maintains information about all trials it has been dealing
    with so far. ``trial_decision`` is the current status of the trial.
    ``keep_case`` is relevant only if ``searcher_data == "rungs_and_last"``.
    ``largest_update_resource`` is the largest resource level for which the
    searcher was updated, or None.
    ``reported_result`` contains the last recent reported result, or None
    (task was started, but did not report anything yet). Only contains
    attributes ``self.metric`` and ``self._resource_attr``.
    """

    config: Dict[str, Any]
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


class HyperbandScheduler(
    FIFOScheduler, MultiFidelitySchedulerMixin, RemoveCheckpointsSchedulerMixin
):
    r"""Implements different variants of asynchronous Hyperband

    See ``type`` for the different variants. One implementation detail is
    when using multiple brackets, task allocation to bracket is done randomly,
    based on a distribution which can be configured.

    For definitions of concepts (bracket, rung, milestone), see

        | Li, Jamieson, Rostamizadeh, Gonina, Hardt, Recht, Talwalkar (2018)
        | A System for Massively Parallel Hyperparameter Tuning
        | https://arxiv.org/abs/1810.05934

    or

        | Tiao, Klein, Lienart, Archambeau, Seeger (2020)
        | Model-based Asynchronous Hyperparameter and Neural Architecture Search
        | https://arxiv.org/abs/2003.10865

    .. note::
       This scheduler requires both ``metric`` and ``resource_attr`` to be
       returned by the reporter. Here, resource values must be positive int.
       If ``resource_attr == "epoch"``, this should be the number of epochs done,
       starting from 1 (not the epoch number, starting from 0).

    **Rung levels and promotion quantiles**

    Rung levels are values of the resource attribute at which stop/go decisions
    are made for jobs, comparing their metric against others at the same level.
    These rung levels (positive, strictly increasing) can be specified via
    ``rung_levels``, the largest must be ``<= max_t``.
    If ``rung_levels`` is not given, they are specified by ``grace_period``
    and ``reduction_factor`` or ``rung_increment``:

    * If :math:`r_{min}` is ``grace_period``, :math:`\eta` is
      ``reduction_factor``, then rung levels are
      :math:`\mathrm{round}(r_{min} \eta^j), j=0, 1, \dots`. This is the default
      choice for successive halving (Hyperband).
    * If ``rung_increment`` is given, but not ``reduction_factor``, then rung
      levels are :math:`r_{min} + j \nu, j=0, 1, \dots`, where :math:`\nu` is
      ``rung_increment``.

    If ``rung_levels`` is given, then ``grace_period``, ``reduction_factor``,
    ``rung_increment`` are ignored. If they are given, a warning is logged.

    The rung levels determine the quantiles to be used in the stop/go
    decisions. If rung levels are :math:`r_j`, define
    :math:`q_j = r_j / r_{j+1}`.
    :math:`q_j` is the promotion quantile at rung level :math:`r_j`. On
    average, a fraction of :math:`q_j` jobs can continue, the remaining ones
    are stopped (or paused). In the default successive halving case, we have
    :math:`q_j = 1/\eta` for all :math:`j`.

    **Cost-aware schedulers or searchers**

    Some schedulers (e.g., ``type == "cost_promotion"``) or searchers may depend
    on cost values (with key ``cost_attr``) reported alongside the target metric.
    For promotion-based scheduling, a trial may pause and resume several times.
    The cost received in ``on_trial_result`` only counts the cost since the last
    resume. We maintain the sum of such costs in :meth:`_cost_offset`, and append
    a new entry to ``result`` in ``on_trial_result`` with the total cost.
    If the evaluation function does not implement checkpointing, once a trial
    is resumed, it has to start from scratch. We detect this in
    ``on_trial_result`` and reset the cost offset to 0 (if the trial runs from
    scratch, the cost reported needs no offset added).

    .. note::
       This process requires ``cost_attr`` to be set

    **Pending evaluations**

    The searcher is notified, by ``searcher.register_pending`` calls, of
    (trial, resource) pairs for which evaluations are running, and a result
    is expected in the future. These pending evaluations can be used by the
    searcher in order to direct sampling elsewhere.

    The choice of pending evaluations depends on ``searcher_data``. If equal
    to "rungs", pending evaluations sit only at rung levels, because
    observations are only used there. In the other cases, pending evaluations
    sit at all resource levels for which observations are obtained. For
    example, if a trial is at rung level :math:`r` and continues towards the
    next rung level :math:`r_{next}`, if ``searcher_data == "rungs"``,
    ``searcher.register_pending`` is called for :math:`r_{next}` only, while for
    other ``searcher_data`` values, pending evaluations are registered for
    :math:`r + 1, r + 2, \dots, r_{next}`.
    However, if in this case, ``register_pending_myopic`` is ``True``, we instead
    call ``searcher.register_pending`` for :math:`r + 1` when each observation is
    obtained (not just at a rung level). This leads to less pending
    evaluations at any one time. On the other hand, when a trial is continued
    at a rung level, we already know it will emit observations up to the next
    rung level, so it seems more "correct" to register all these pending
    evaluations in one go.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`:

    :param searcher: Searcher for ``get_config`` decisions. String values
        are passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory` along
        with ``search_options`` and extra information. Supported values:
        :const:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.SUPPORTED_SEARCHERS_HYPERBAND`.
        Defaults to "random" (i.e., random search)
    :type searcher: str or
        :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher`
    :param resource_attr: Name of resource attribute in results obtained
        via ``on_trial_result``, defaults to "epoch"
    :type resource_attr: str, optional
    :param grace_period: Minimum resource to be used for a job. Ignored
        if ``rung_levels`` is given. Defaults to 1
    :type grace_period: int, optional
    :param reduction_factor: Parameter to determine rung levels. Ignored
        if ``rung_levels`` is given. Must be :math:`\ge 2`, defaults to 3
    :type reduction_factor: float, optional
    :param rung_increment: Parameter to determine rung levels. Ignored
        if ``rung_levels`` or ``reduction_factor`` are given. Must be
        postive
    :type rung_increment: int, optional
    :param rung_levels: If given, prescribes the set of rung levels to
        be used. Must contain positive integers, strictly increasing.
        This information overrides ``grace_period``, ``reduction_factor``,
        ``rung_increment``. Note that the stop/promote rule in the successive
        halving scheduler is set based on the ratio of successive rung levels.
    :type rung_levels: ``List[int]``, optional
    :param brackets: Number of brackets to be used in Hyperband. Each
        bracket has a different grace period, all share ``max_t``
        and ``reduction_factor``. If ``brackets == 1`` (default), we run
        asynchronous successive halving.
    :type brackets: int, optional
    :param type: Type of Hyperband scheduler. Defaults to "stopping".
        Supported values (see also subclasses of
        :class:`~syne_tune.optimizer.schedulers.hyperband_stopping.RungSystem`):

        * stopping: A config eval is executed by a single task. The task is
          stopped at a milestone if its metric is worse than a fraction
          of those who reached the milestone earlier, otherwise it
          continues. See
          :class:`~syne_tune.optimizer.schedulers.hyperband_stopping.StoppingRungSystem`.
        * promotion: A config eval may be associated with multiple tasks
          over its lifetime. It is never terminated, but may be paused.
          Whenever a task becomes available, it may promote a config to
          the next milestone, if better than a fraction of others who
          reached the milestone. If no config can be promoted, a new one
          is chosen. See
          :class:`~syne_tune.optimizer.schedulers.hyperband_promotion.PromotionRungSystem`.
        * cost_promotion: This is a cost-aware variant of 'promotion', see
          :class:`~syne_tune.optimizer.schedulers.hyperband_cost_promotion.CostPromotionRungSystem`
          for details. In this case, costs must be reported under the name
          ``rung_system_kwargs["cost_attr"]`` in results.
        * pasha: Similar to promotion type Hyperband, but it progressively
          expands the available resources until the ranking of
          configurations stabilizes.
        * rush_stopping: A variation of the stopping scheduler which requires
          passing ``rung_system_kwargs`` and ``points_to_evaluate``. The first
          ``rung_system_kwargs["num_threshold_candidates"]`` of
          ``points_to_evaluate`` will enforce stricter rules on which task is
          continued. See
          :class:`~syne_tune.optimizer.schedulers.hyperband_rush.RUSHStoppingRungSystem`
          and
          :class:`~syne_tune.optimizer.schedulers.transfer_learning.RUSHScheduler`.
        * rush_promotion: Same as ``rush_stopping`` but for promotion, see
          :class:`~syne_tune.optimizer.schedulers.hyperband_rush.RUSHPromotionRungSystem`
        * dyhpo: A model-based scheduler, which can be seen as extension of
          "promotion" with ``rung_increment`` rather than ``reduction_factor``, see
          :class:`~syne_tune.optimizer.schedulers.searchers.dyhpo.DynamicHPOSearcher`

    :type type: str, optional
    :param cost_attr: Required if the scheduler itself uses a cost metric
        (i.e., ``type="cost_promotion"``), or if the searcher uses a cost
        metric. See also header comment.
    :type cost_attr: str, optional
    :param searcher_data: Relevant only if a model-based searcher is used.
        Example: For NN tuning and ``resource_attr == "epoch"', we receive a
        result for each epoch, but not all epoch values are also rung levels.
        searcher_data determines which of these results are passed to the
        searcher. As a rule, the more data the searcher receives, the better
        its fit, but also the more expensive get_config may become. Choices:

        * "rungs" (default): Only results at rung levels. Cheapest
        * "all": All results. Most expensive
        * "rungs_and_last": Results at rung levels, plus the most recent
          result. This means that in between rung levels, only the most
          recent result is used by the searcher. This is in between

        Note: For a Gaussian additive learning curve surrogate model, this
        has to be set to 'all'.
    :type searcher_data: str, optional
    :param register_pending_myopic: See above. Used only if ``searcher_data !=
        "rungs"``. Defaults to ``False``
    :type register_pending_myopic: bool, optional
    :param rung_system_per_bracket: This concerns Hyperband with
        ``brackets > 1``. Defaults to ``False``.
        When starting a job for a new config, it is assigned a randomly
        sampled bracket. The larger the bracket, the larger the grace period
        for the config.
        If ``rung_system_per_bracket == True``, we maintain separate rung level
        systems for each bracket, so that configs only compete with others
        started in the same bracket.
        If ``rung_system_per_bracket == False``, we use a single rung level system,
        so that all configs compete with each other. In this case, the bracket
        of a config only determines the initial grace period, i.e. the first
        milestone at which it starts competing with others. This is the
        default.
        The concept of brackets in Hyperband is meant to hedge against overly
        aggressive filtering in successive halving, based on low fidelity
        criteria. In practice, successive halving (i.e., ``brackets = 1``) often
        works best in the asynchronous case (as implemented here). If
        ``brackets > 1``, the hedging is stronger if ``rung_system_per_bracket``
        is ``True``.
    :type rung_system_per_bracket: bool, optional
    :param do_snapshots: Support snapshots? If ``True``, a snapshot of all running
        tasks and rung levels is returned by :meth:`_promote_trial`. This
        snapshot is passed to ``searcher.get_config``. Defaults to ``False``.
        Note: Currently, only the stopping variant supports snapshots.
    :type do_snapshots: bool, optional
    :param rung_system_kwargs: Arguments passed to the rung system:
        * num_threshold_candidates: Used if ``type in ["rush_promotion",
          "rush_stopping"]``. The first ``num_threshold_candidates`` in
          ``points_to_evaluate`` enforce stricter requirements to the
          continuation of training tasks. See
          :class:`~syne_tune.optimizer.schedulers.transfer_learning.RUSHScheduler`.
        * probability_sh: Used if ``type == "dyhpo"``. In DyHPO, we typically
          all paused trials against a number of new configurations, and the
          winner is either resumed or started (new trial). However, with the
          probability given here, we instead try to promote a trial as if
          ``type == "promotion"``. If no trial can be promoted, we fall back to
          the DyHPO logic. Use this to make DyHPO robust against starting too
          many new trials, because all paused ones score poorly (this happens
          especially at the beginning).

    :type rung_system_kwargs: Dict[str, Any], optional
    :param early_checkpoint_removal_kwargs: If given, speculative early removal
        of checkpoints is done, see
        :class:`~syne_tune.callbacks.hyperband_remove_checkpoints_callback.HyperbandRemoveCheckpointsCallback`.
        The constructor arguments for the ``HyperbandRemoveCheckpointsCallback``
        must be given here, if they cannot be inferred (key ``max_num_checkpoints``
        is mandatory). This feature is used only for scheduler types which pause
        and resume trials.
    :type early_checkpoint_removal_kwargs: Dict[str, Any], optional
    """

    def __init__(self, config_space: Dict[str, Any], **kwargs):
        # Before we can call the superclass constructor, we need to set a few
        # members (see also ``_extend_search_options``).
        # To do this properly, we first check values and impute defaults for
        # ``kwargs``.
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
        if (
            kwargs.get("reduction_factor") is None
            and kwargs.get("rung_increment") is None
        ):
            kwargs["reduction_factor"] = 3
        scheduler_type = kwargs["type"]
        self.scheduler_type = scheduler_type
        self._resource_attr = kwargs["resource_attr"]
        self._rung_system_kwargs = kwargs["rung_system_kwargs"]
        self._cost_attr = kwargs.get("cost_attr")
        self._searcher_data = kwargs["searcher_data"]
        assert not (
            scheduler_type == "cost_promotion" and self._cost_attr is None
        ), "cost_attr must be given if type='cost_promotion'"
        # Default: May be modified by searcher (via ``configure_scheduler``)
        self.bracket_distribution = DefaultHyperbandBracketDistribution()
        # See :meth:`_extend_search_options` for why this is needed:
        if kwargs.get("searcher") == "hypertune":
            self._num_brackets_info = {
                "num_brackets": kwargs["brackets"],
                "rung_levels": kwargs.get("rung_levels"),
                "grace_period": kwargs["grace_period"],
                "reduction_factor": kwargs.get("reduction_factor"),
                "rung_increment": kwargs.get("rung_increment"),
            }
        else:
            self._num_brackets_info = None
        if self.does_pause_resume() and "max_resource_attr" not in kwargs:
            logger.warning(
                "You do not specify max_resource_attr, but use max_t instead. "
                "This is not recommended best practice and may lead to a loss "
                "of efficiency. Consider using max_resource_attr instead.\n"
                "See https://syne-tune.readthedocs.io/en/latest/tutorials/multifidelity/mf_setup.html#the-launcher-script "
                "for details."
            )

        # Superclass constructor
        super().__init__(config_space, **filter_by_key(kwargs, _ARGUMENT_KEYS))
        assert self.max_t is not None, (
            "Either max_t must be specified, or it has to be specified as "
            + "config_space[max_resource_attr], config_space['epochs'], "
            + "config_space['max_t'], config_space['max_epochs']"
        )

        # If rung_levels is given, grace_period and reduction_factor are ignored
        rung_levels = kwargs.get("rung_levels")
        if rung_levels is not None:
            assert isinstance(rung_levels, list)
            if (
                ("grace_period" in kwargs)
                or ("reduction_factor" in kwargs)
                or ("rung_increment" in kwargs)
            ):
                logger.warning(
                    "Since rung_levels is given, the values grace_period = "
                    f"{kwargs.get('grace_period')}, reduction_factor = "
                    f"{kwargs.get('reduction_factor')} and rung_increment = "
                    f"{kwargs.get('rung_increment')} are ignored!"
                )
        rung_levels = successive_halving_rung_levels(
            rung_levels,
            grace_period=kwargs["grace_period"],
            reduction_factor=kwargs.get("reduction_factor"),
            rung_increment=kwargs.get("rung_increment"),
            max_t=self.max_t,
        )
        do_snapshots = kwargs["do_snapshots"]
        assert (not do_snapshots) or (
            scheduler_type == "stopping"
        ), "Snapshots are supported only for type = 'stopping'"
        rung_system_per_bracket = kwargs["rung_system_per_bracket"]

        self.terminator = HyperbandBracketManager(
            scheduler_type=scheduler_type,
            resource_attr=self._resource_attr,
            metric=self.metric,
            mode=self.mode,
            max_t=self.max_t,
            rung_levels=rung_levels,
            brackets=kwargs["brackets"],
            rung_system_per_bracket=rung_system_per_bracket,
            cost_attr=self._total_cost_attr(),
            random_seed=self.random_seed_generator(),
            rung_system_kwargs=self._rung_system_kwargs,
            scheduler=self,
        )
        self.do_snapshots = do_snapshots
        self._register_pending_myopic = kwargs["register_pending_myopic"]
        # _active_trials:
        # Maintains information for all tasks (running, paused, or stopped).
        # Maps trial_id to ``TrialInformation``.
        self._active_trials = dict()
        # _cost_offset:
        # Is used for promotion-based (pause/resume) scheduling if the eval
        # function reports cost values. For a trial which has been paused at
        # least once, this records the sum of costs for reaching its last
        # recent milestone.
        self._cost_offset = dict()
        self._initialize_early_checkpoint_removal(
            kwargs.get("early_checkpoint_removal_kwargs")
        )

    def _initialize_early_checkpoint_removal(
        self, callback_kwargs: Optional[Dict[str, Any]]
    ):
        if callback_kwargs is not None:
            for name in ["max_num_checkpoints"]:
                assert (
                    name in callback_kwargs
                ), f"early_checkpoint_removal_kwargs must contain '{name}' entry"
        self._early_checkpoint_removal_kwargs = callback_kwargs

    def does_pause_resume(self) -> bool:
        """
        :return: Is this variant doing pause and resume scheduling, in the
            sense that trials can be paused and resumed later?
        """
        return HyperbandBracketManager.does_pause_resume(self.scheduler_type)

    @property
    def rung_levels(self) -> List[int]:
        """
        Note that all entries of ``rung_levels`` are smaller than ``max_t`` (or
        ``config_space[max_resource_attr]``): rung levels are resource levels where
        stop/go decisions are made. In particular, if ``rung_levels`` is passed at
        construction with ``rung_levels[-1] == max_t``, this last entry is stripped
        off.

        :return: Rung levels (strictly increasing, positive ints)
        """
        return self.terminator.rung_levels

    @property
    def num_brackets(self) -> int:
        return self.terminator.num_brackets

    @property
    def resource_attr(self) -> str:
        return self._resource_attr

    @property
    def max_resource_level(self) -> int:
        return self.max_t

    @property
    def searcher_data(self) -> str:
        return self._searcher_data

    def _initialize_searcher(self):
        if not self._searcher_initialized:
            super()._initialize_searcher()
            self.bracket_distribution.configure(self)

    def _extend_search_options(self, search_options: Dict[str, Any]) -> Dict[str, Any]:
        # Note: Needs ``self.scheduler_type`` to be set
        scheduler = "hyperband_{}".format(self.scheduler_type)
        result = dict(
            search_options,
            scheduler=scheduler,
            resource_attr=self._resource_attr,
            searcher_data=self._searcher_data,
        )
        # Cost attribute: For promotion-based, cost needs to be accumulated
        # for each trial
        cost_attr = self._total_cost_attr()
        if cost_attr is not None:
            result["cost_attr"] = cost_attr
        if self._num_brackets_info is not None:
            # At this point, the correct number of brackets needs to be
            # determined. This could be smaller than the ``brackets`` argument
            # passed at construction, since the number of brackets is limited
            # by the number of rung levels, which in turn requires ``max_t``
            # to have been determined. This is why we need some extra effort
            # here
            rung_levels = successive_halving_rung_levels(
                self._num_brackets_info["rung_levels"],
                grace_period=self._num_brackets_info["grace_period"],
                reduction_factor=self._num_brackets_info["reduction_factor"],
                rung_increment=self._num_brackets_info["rung_increment"],
                max_t=self.max_t,
            )
            num_brackets = min(
                self._num_brackets_info["num_brackets"], len(rung_levels) + 1
            )
            result["hypertune_distribution_num_brackets"] = num_brackets
        return result

    def _total_cost_attr(self) -> Optional[str]:
        """
        In pause and resume scheduling, the total cost for a trial so far is
        the sum of costs over all jobs associated with the trial.

        :return: Name of attribute for total cost
        """
        if self._cost_attr is None:
            return None
        elif self.does_pause_resume():
            return "total_" + self._cost_attr
        else:
            return self._cost_attr

    def _on_config_suggest(
        self, config: Dict[str, Any], trial_id: str, **kwargs
    ) -> Dict[str, Any]:
        """
        ``kwargs`` being used here:

        * elapsed_time: Time from start of experiment, set in
          :meth:`~syne_tune.optimizer.schedulers.FIFOScheduler._suggest`
        * bracket: Bracket in which new trial is started, set in
          :meth:`~syne_tune.optimizer.schedulers.HyperbandScheduler._promote_trial`
        * milestone: First milestone the new trial will reach, set in
          :meth:`~syne_tune.optimizer.schedulers.HyperbandScheduler._promote_trial`

        :param config: New config suggested for ``trial_id``
        :param trial_id: Input to ``_suggest``
        :param kwargs: Optional. Additional args
        :return: Configuration, potentially modified
        """
        assert trial_id not in self._active_trials, f"Trial {trial_id} already exists"
        # See ``FIFOScheduler._on_config_suggest`` for why we register the task
        # and pending evaluation here, and not later in ``on_task_add``.
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
        # Extra fields in ``config``
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
    def _promote_trial(self, new_trial_id: str) -> (Optional[str], Optional[dict]):
        """
        If ``self.do_snapshots``, a snapshot is written to
        ``extra_kwargs["snapshot"]``:

        * max_resource
        * reduction_factor
        * tasks: Info about running tasks in bracket bracket_id (or, if
          brackets share the same rung level system, all running tasks):
          ``dict(task_id) -> dict``:
        * config: ``config`` as dict
        * time: Time when task was started, or when last recent result was
         reported
        * level: Level of last recent result report, or 0 if no reports yet
        * rungs: Metric values at rung levels in bracket bracket_id:
          List of ``(rung_level, metric_dict)``, where ``metric_dict`` has entries
          :code:`task_id: metric_value`. Note that entries are sorted in
          decreasing order w.r.t. ``rung_level``.

        :return: ``(trial_id, extra_kwargs)``
        """
        trial_id, extra_kwargs = self.terminator.on_task_schedule(new_trial_id)
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

        :param trial_id: ID of trial
        :param trial_decision: Decision taken for this trial
        """
        self.terminator.on_task_remove(trial_id)
        if trial_id in self._active_trials:
            # We do not remove stopped trials
            self._active_trials[trial_id].trial_decision = trial_decision

    def on_trial_error(self, trial: Trial):
        super().on_trial_error(trial)
        self._cleanup_trial(str(trial.trial_id), trial_decision=SchedulerDecision.STOP)

    def _update_searcher_internal(
        self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]
    ):
        if self.searcher_data == "rungs_and_last":
            # Remove last recently added result for this task. This is not
            # done if it fell on a rung level (i.e., ``keep_case`` is True)
            record = self._active_trials[trial_id]
            rem_result = record.reported_result
            if (rem_result is not None) and (not record.keep_case):
                self.searcher.remove_case(trial_id, **rem_result)

    def _update_searcher(
        self,
        trial_id: str,
        config: Dict[str, Any],
        result: Dict[str, Any],
        task_info: Dict[str, Any],
    ):
        """Updates searcher with ``result``, registers pending config there

        :param trial_id: ID of trial
        :param config: Configuration for trial
        :param result: Record obtained from ``on_trial_result``
        :param task_info: Info from ``self.terminator.on_task_report``
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
                # Note: This condition is weaker than ``milestone_reached`` if
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
                    # ``next_milestone``
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

    def _check_result(self, result: Dict[str, Any]):
        super()._check_result(result)
        keys = [self._resource_attr]
        if self.scheduler_type == "cost_promotion":
            keys.append(self._cost_attr)
        self._check_keys_of_result(result, keys)
        resource = result[self._resource_attr]
        assert 1 <= resource == round(resource), (
            "Your training evaluation function needs to report positive "
            + f"integer values for key {self._resource_attr}. Obtained "
            + f"value {resource}, which is not permitted"
        )

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        self._check_result(result)
        trial_id = str(trial.trial_id)
        debug_log = self.searcher.debug_log
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
        ignore_data = False
        if trial_decision != SchedulerDecision.CONTINUE:
            logger.warning(
                f"trial_id {trial_id}: {trial_decision}, but receives "
                f"another report {result}\nThis report is ignored"
            )
        else:
            task_info = self.terminator.on_task_report(trial_id, result)
            task_continues = task_info["task_continues"]
            milestone_reached = task_info["milestone_reached"]
            ignore_data = task_info.get("ignore_data", False)
            if cost_and_promotion:
                if milestone_reached:
                    # Trial reached milestone and will pause there: Update
                    # cost offset
                    if self._cost_attr is not None:
                        self._cost_offset[trial_id] = result[self._total_cost_attr()]
                elif ignore_data:
                    # For a resumed trial, the report is for resource <=
                    # resume_from, where resume_from < milestone. This
                    # happens if checkpointing is not implemented and a
                    # resumed trial has to start from scratch, publishing
                    # results all the way up to resume_from. In this case,
                    # we can erase the ``_cost_offset`` entry, since the
                    # instantaneous cost reported by the trial does not
                    # have any offset.
                    if self._cost_offset[trial_id] > 0:
                        logger.info(
                            f"trial_id {trial_id}: Resumed trial seems to have been "
                            + "started from scratch (no checkpointing?), so we erase "
                            + "the cost offset."
                        )
                    self._cost_offset[trial_id] = 0

            if not ignore_data:
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
                act_str = None
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

        if not ignore_data:
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

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        # Check whether searcher was already updated based on ``result``
        trial_id = str(trial.trial_id)
        largest_update_resource = self._active_trials[trial_id].largest_update_resource
        if largest_update_resource is not None:
            resource = int(result[self._resource_attr])
            if resource > largest_update_resource:
                super().on_trial_complete(trial, result)
        # Remove pending evaluations, in case there are still some
        self.searcher.cleanup_pending(trial_id)
        self._cleanup_trial(trial_id, trial_decision=SchedulerDecision.STOP)

    def callback_for_checkpoint_removal(
        self, stop_criterion: Callable[[TuningStatus], bool]
    ) -> Optional[TunerCallback]:
        if (
            self._early_checkpoint_removal_kwargs is None
            or not self.terminator.support_early_checkpoint_removal()
        ):
            return None
        else:
            callback_kwargs = dict(
                self._early_checkpoint_removal_kwargs,
                metric=self.metric,
                resource_attr=self._resource_attr,
                mode=self.mode,
            )
            return create_callback_for_checkpoint_removal(
                callback_kwargs, stop_criterion=stop_criterion
            )


class HyperbandBracketManager:
    """
    Maintains rung level systems for range of brackets. Differences depending
    on ``scheduler_type`` manifest themselves mostly at the level of the rung
    level system itself.

    :param scheduler_type: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
    :param resource_attr: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
    :param metric: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
    :param mode: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
    :param max_t: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
    :param rung_levels: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
    :param brackets: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
    :param rung_system_per_bracket: See :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`.
    :param cost_attr: Overrides entry in ``rung_system_kwargs``
    :param random_seed: Random seed for bracket sampling
    :param rung_system_kwargs: Arguments passed to the rung system
    :param scheduler: The scheduler is needed in order to sample a bracket, and
        also some rung level systems need more information from the scheduler
    """

    def __init__(
        self,
        scheduler_type: str,
        resource_attr: str,
        metric: str,
        mode: str,
        max_t: int,
        rung_levels: List[int],
        brackets: int,
        rung_system_per_bracket: bool,
        cost_attr: str,
        random_seed: int,
        rung_system_kwargs: Dict[str, Any],
        scheduler: HyperbandScheduler,
    ):
        assert (
            rung_levels[-1] < max_t
        ), f"rung_levels = {rung_levels} must not contain max_t = {max_t}"
        self._scheduler_type = scheduler_type
        self._resource_attr = resource_attr
        self._max_t = max_t
        self.rung_levels = copy.copy(rung_levels)
        self._rung_system_per_bracket = rung_system_per_bracket
        self._scheduler = scheduler
        self.random_state = np.random.RandomState(random_seed)
        # Maps trial_id -> bracket_id
        self._task_info = dict()
        max_num_brackets = len(rung_levels) + 1
        self.num_brackets = min(brackets, max_num_brackets)
        num_systems = self.num_brackets if rung_system_per_bracket else 1
        rung_levels_plus_maxt = rung_levels[1:] + [max_t]
        # Promotion quantiles: q_j = r_j / r_{j+1}
        promote_quantiles = [x / y for x, y in zip(rung_levels, rung_levels_plus_maxt)]
        kwargs = dict(
            metric=metric,
            mode=mode,
            resource_attr=resource_attr,
            max_t=max_t,
        )
        if scheduler_type in ["rush_promotion", "rush_stopping"]:
            kwargs["num_threshold_candidates"] = rung_system_kwargs.get(
                "num_threshold_candidates", 0
            )
        elif scheduler_type == "cost_promotion":
            kwargs["cost_attr"] = cost_attr
        elif scheduler_type == "dyhpo":
            kwargs["searcher"] = scheduler.searcher
            kwargs["probability_sh"] = rung_system_kwargs["probability_sh"]
            kwargs["random_state"] = self.random_state
        rs_type = RUNG_SYSTEMS[scheduler_type]
        self._rung_systems = [
            rs_type(
                rung_levels=rung_levels[s:],
                promote_quantiles=promote_quantiles[s:],
                **kwargs,
            )
            for s in range(num_systems)
        ]

    @staticmethod
    def does_pause_resume(scheduler_type: str) -> bool:
        """
        :return: Is this variant doing pause and resume scheduling, in the
            sense that trials can be paused and resumed later?
        """
        return RUNG_SYSTEMS[scheduler_type].does_pause_resume()

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
        (decreasing) order. The first entry is ``max_t``, even if it is
        not a rung level in the bracket. This list contains the resource
        levels the task would reach if it ran to ``max_t`` without being stopped.

        :param trial_id: ID of trial
        :param kwargs: Further arguments passed to ``rung_sys.on_task_add``
        :return: List of milestones in decreasing order, where`` max_t`` is first
        """
        assert "bracket" in kwargs
        bracket_id = kwargs["bracket"]
        self._task_info[trial_id] = bracket_id
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        rung_sys.on_task_add(trial_id, skip_rungs=skip_rungs, **kwargs)
        milestones = rung_sys.get_milestones(skip_rungs)
        if milestones:
            milestones.insert(0, self._max_t)
        else:
            milestones = [self._max_t]
        return milestones

    def on_task_report(self, trial_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method is called whenever a new report is received. It returns a
        dictionary with all the information needed for making decisions
        (e.g., stop / continue task, update model, etc). Keys are:

        * ``task_continues``: Should task continue or stop/pause?
        * ``milestone_reached``: True if rung level (or ``max_t``) is hit
        * ``next_milestone``: If hit ``rung level < max_t``, this is the subsequent
          rung level (otherwise: None)
        * ``bracket_id``: Bracket in which the task is running

        :param trial_id: ID of trial
        :param result: Results reported
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
        """Called when trial is stopped or completes

        :param trial_id: ID of trial
        """
        if trial_id in self._task_info:
            rung_sys, _, _ = self._get_rung_system(trial_id)
            rung_sys.on_task_remove(trial_id)
            del self._task_info[trial_id]

    def _sample_bracket(self) -> int:
        """Samples bracket number from bracket distribution

        The bracket distribution is provided by the scheduler. For certain
        child classes, it can be adaptive.

        :return: Bracket number sampled from distribution
        """
        distribution = self._scheduler.bracket_distribution()
        return self.random_state.choice(a=distribution.size, p=distribution)

    def on_task_schedule(self, new_trial_id: str) -> (Optional[str], dict):
        """
        Samples bracket for task to be scheduled. Check whether any paused
        trial in that bracket can be promoted. If so, its ``trial_id`` is
        returned. We also return ``extra_kwargs`` to be used in ``_promote_trial``.
        This contains the bracket which was sampled (key "bracket").

        Note: ``extra_kwargs`` can return information also if ``trial_id = None``
        is returned. This information is passed to ``get_config`` of the
        searcher.

        Note: ``extra_kwargs`` can return information also if ``trial_id = None``
        is returned. This information is passed to ``get_config`` of the
        searcher.

        :param new_trial_id: ID for new trial as passed to :meth:`_suggest`
        :return: ``(trial_id, extra_kwargs)``
        """
        # Sample bracket for task to be scheduled
        bracket_id = self._sample_bracket()
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        extra_kwargs = {"bracket": bracket_id}
        # Check whether config can be promoted
        ret_dict = rung_sys.on_task_schedule(new_trial_id)
        trial_id = ret_dict.get("trial_id")
        if trial_id is not None:
            del ret_dict["trial_id"]
        extra_kwargs.update(ret_dict)
        k = "milestone"
        if k not in extra_kwargs:
            extra_kwargs[k] = rung_sys.get_first_milestone(skip_rungs)
        return trial_id, extra_kwargs

    def snapshot_rungs(self, bracket_id):
        rung_sys, skip_rungs = self._get_rung_system_for_bracket_id(bracket_id)
        return rung_sys.snapshot_rungs(skip_rungs)

    def paused_trials(self, resource: Optional[int] = None) -> PausedTrialsResult:
        """
        Only for pause and resume schedulers (:meth:`does_pause_resume` returns
        ``True``), where trials can be paused at certain rung levels only.
        If ``resource`` is not given, returns list of all paused trials
        ``(trial_id, rank, metric_val, level)``, where ``level`` is
        the rung level, and ``rank`` is the rank of the trial in the rung
        (0 for the best metric value). If ``resource`` is given, only the
        paused trials in the rung of this level are returned.

        :param resource: If given, paused trials of only this rung level are
            returned. Otherwise, all paused trials are returned
        :return: See above
        """
        return [
            entry for rs in self._rung_systems for entry in rs.paused_trials(resource)
        ]

    def information_for_rungs(self) -> List[Tuple[int, int, float]]:
        """
        :return: List of ``(resource, num_entries, prom_quant)``, where
            ``resource`` is a rung level, ``num_entries`` the number of entries
            in the rung, and ``prom_quant`` the promotion quantile
        """
        return self._rung_systems[0].information_for_rungs()

    def support_early_checkpoint_removal(self) -> bool:
        return self._rung_systems[0].support_early_checkpoint_removal()
