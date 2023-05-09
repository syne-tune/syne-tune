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
from typing import Optional, List, Set, Dict, Any, Tuple
import logging
import numpy as np

from syne_tune.callbacks.remove_checkpoints_callback import (
    DefaultRemoveCheckpointsSchedulerMixin,
)
from syne_tune.optimizer.schedulers.random_seeds import RANDOM_SEED_UPPER_BOUND
from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket_manager import (
    SynchronousHyperbandBracketManager,
)
from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket import SlotInRung
from syne_tune.optimizer.schedulers.synchronous.hyperband_rung_system import (
    RungSystemsPerBracket,
)
from syne_tune.optimizer.scheduler import TrialSuggestion, SchedulerDecision
from syne_tune.optimizer.schedulers.scheduler_searcher import TrialSchedulerWithSearcher
from syne_tune.optimizer.schedulers.multi_fidelity import MultiFidelitySchedulerMixin
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import cast_config_values
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    Categorical,
    String,
    assert_no_invalid_options,
    Integer,
)
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_factory

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    "searcher",
    "search_options",
    "metric",
    "mode",
    "points_to_evaluate",
    "random_seed",
    "max_resource_attr",
    "max_resource_level",
    "resource_attr",
    "searcher_data",
}

_DEFAULT_OPTIONS = {
    "searcher": "random",
    "mode": "min",
    "resource_attr": "epoch",
    "searcher_data": "rungs",
}

_CONSTRAINTS = {
    "metric": String(),
    "mode": Categorical(choices=("min", "max")),
    "random_seed": Integer(0, RANDOM_SEED_UPPER_BOUND),
    "max_resource_attr": String(),
    "max_resource_level": Integer(1, None),
    "resource_attr": String(),
    "searcher_data": Categorical(("rungs", "all")),
}


class SynchronousHyperbandCommon(
    TrialSchedulerWithSearcher, MultiFidelitySchedulerMixin
):
    """
    Common code for :meth:`_create_internal` in
    :class:`~syne_tune.optimizer.schedulers.synchronous.SynchronousHyperbandScheduler`
    and
    :class:`~syne_tune.optimizer.schedulers.synchronous.DifferentialEvolutionHyperbandScheduler`
    """

    def _create_internal_common(
        self, skip_searchers: Optional[Set[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        self.metric = kwargs.get("metric")
        assert self.metric is not None, (
            "Argument 'metric' is mandatory. Pass the name of the metric "
            + "reported by your training script, which you'd like to "
            + "optimize, and use 'mode' to specify whether it should "
            + "be minimized or maximized"
        )
        self.mode = kwargs["mode"]
        self.max_resource_attr = kwargs.get("max_resource_attr")
        self._resource_attr = kwargs["resource_attr"]
        if self.max_resource_attr is None:
            logger.warning(
                "You do not specify max_resource_attr, but use max_resource_level "
                "instead. This is not recommended best practice and may lead to a "
                "loss of efficiency. Consider using max_resource_attr instead.\n"
                "See https://syne-tune.readthedocs.io/en/latest/tutorials/multifidelity/mf_setup.html#the-launcher-script "
                "for details."
            )
        self._max_resource_level = self._infer_max_resource_level(
            max_resource_level=kwargs.get("max_resource_level"),
            max_resource_attr=self.max_resource_attr,
        )
        assert self._max_resource_level is not None, (
            "Maximum resource level has to be specified, please provide "
            "max_resource_attr or max_resource_level argument."
        )
        self._searcher_data = kwargs["searcher_data"]
        # Generate searcher
        searcher = kwargs["searcher"]
        assert isinstance(
            searcher, str
        ), f"searcher must be of type string, but has type {type(searcher)}"
        search_options = kwargs.get("search_options")
        if search_options is None:
            search_options = dict()
        else:
            search_options = search_options.copy()
        if skip_searchers is None or searcher not in skip_searchers:
            search_options.update(
                {
                    "config_space": self.config_space.copy(),
                    "metric": self.metric,
                    "points_to_evaluate": kwargs.get("points_to_evaluate"),
                    "mode": kwargs["mode"],
                    "random_seed_generator": self.random_seed_generator,
                    "resource_attr": self._resource_attr,
                    "scheduler": "hyperband_synchronous",
                }
            )
            if searcher == "bayesopt":
                search_options["max_epochs"] = self._max_resource_level
            self._searcher: BaseSearcher = searcher_factory(searcher, **search_options)
        else:
            self._searcher = None
        return search_options

    @property
    def searcher(self) -> Optional[BaseSearcher]:
        return self._searcher

    @property
    def resource_attr(self) -> str:
        return self._resource_attr

    @property
    def max_resource_level(self) -> int:
        return self._max_resource_level

    @property
    def searcher_data(self) -> str:
        return self._searcher_data


class SynchronousHyperbandScheduler(
    SynchronousHyperbandCommon, DefaultRemoveCheckpointsSchedulerMixin
):
    """
    Synchronous Hyperband. Compared to
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler`, this is also
    scheduling jobs asynchronously, but decision-making is synchronized,
    in that trials are only promoted to the next milestone once the rung they
    are currently paused at, is completely occupied.

    Our implementation never delays scheduling of a job. If the currently
    active bracket does not accept jobs, we assign the job to a later bracket.
    This means that at any point in time, several brackets can be active, but
    jobs are preferentially assigned to the first one (the "primary" active
    bracket).

    :param config_space: Configuration space for trial evaluation function
    :param bracket_rungs: Determines rung level systems for each bracket, see
        :class:`~syne_tune.optimizer.schedulers.synchronous.hyperband_bracket_manager.SynchronousHyperbandBracketManager`
    :param metric: Name of metric to optimize, key in result's obtained via
        :meth:`on_trial_result`
    :type metric: str
    :param searcher: Searcher for ``get_config`` decisions. Passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory` along
        with ``search_options`` and extra information. Supported values:
        :const:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.SUPPORTED_SEARCHERS_HYPERBAND`.
        Defaults to "random" (i.e., random search)
    :type searcher: str, optional
    :param search_options: Passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory`.
    :type search_options: Dict[str, Any], optional
    :param mode: Mode to use for the metric given, can be "min" (default) or
        "max"
    :type mode: str, optional
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If None (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    :type points_to_evaluate: ``List[dict]``, optional
    :param random_seed: Master random seed. Generators used in the scheduler
        or searcher are seeded using
        :class:`~syne_tune.optimizer.schedulers.random_seeds.RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    :param max_resource_attr: Key name in config for fixed attribute
        containing the maximum resource. If given, trials need not be
        stopped, which can run more efficiently.
    :type max_resource_attr: str, optional
    :param max_resource_level: Largest rung level, corresponds to ``max_t`` in
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. Must be positive
        int larger than ``grace_period``. If this is not given, it is inferred
        like in :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. In
        particular, it is not needed if ``max_resource_attr`` is given.
    :type max_resource_level: int, optional
    :param resource_attr: Name of resource attribute in results obtained via
        ``:meth:`on_trial_result`. The type of resource must be int. Default to
        "epoch"
    :type resource_attr: str, optional
    :param searcher_data: Relevant only if a model-based searcher is used.
        Example: For NN tuning and ``resource_attr == "epoch"``, we receive a
        result for each epoch, but not all epoch values are also rung levels.
        searcher_data determines which of these results are passed to the
        searcher. As a rule, the more data the searcher receives, the better
        its fit, but also the more expensive get_config may become. Choices:

        * "rungs" (default): Only results at rung levels. Cheapest
        * "all": All results. Most expensive

        Note: For a Gaussian additive learning curve surrogate model, this
        has to be set to "all".
    :type searcher_data: str, optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        bracket_rungs: RungSystemsPerBracket,
        **kwargs,
    ):
        super().__init__(config_space, **kwargs)
        self._create_internal(bracket_rungs, **kwargs)

    def _create_internal(self, bracket_rungs: RungSystemsPerBracket, **kwargs):
        # Check values and impute default values
        assert_no_invalid_options(
            kwargs, _ARGUMENT_KEYS, name="SynchronousHyperbandScheduler"
        )
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
        self._create_internal_common(**kwargs)
        # Bracket manager
        self.bracket_manager = SynchronousHyperbandBracketManager(
            bracket_rungs,
            mode=self.mode,
        )
        # Maps trial_id to tuples (bracket_id, slot_in_rung), as returned
        # by ``bracket_manager.next_job``, and required by
        # ``bracket_manager.on_result``. Entries are removed once passed to
        # ``on_result``. Note that a trial_id can be associated with different
        # job descriptions in its lifetime
        self._trial_to_pending_slot = dict()
        # Maps trial_id (active) to config
        self._trial_to_config = dict()
        self._rung_levels = [level for _, level in bracket_rungs[0]]
        self._trials_checkpoints_can_be_removed = []

    @property
    def rung_levels(self) -> List[int]:
        return self._rung_levels

    @property
    def num_brackets(self) -> int:
        return len(self.bracket_manager.bracket_rungs)

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        do_debug_log = self.searcher.debug_log is not None
        if do_debug_log and trial_id == 0:
            # This is printed at the start of the experiment. Cannot do this
            # at construction, because with ``RemoteLauncher`` this does not end
            # up in the right log
            parts = ["Rung systems for each bracket:"] + [
                f"Bracket {bracket}: {rungs}"
                for bracket, rungs in enumerate(self.bracket_manager.bracket_rungs)
            ]
            logger.info("\n".join(parts))
        # Ask bracket manager for job
        bracket_id, slot_in_rung = self.bracket_manager.next_job()
        suggestion = None
        if slot_in_rung.trial_id is not None:
            # Paused trial to be resumed (``trial_id`` passed in is ignored)
            trial_id = slot_in_rung.trial_id
            _config = self._trial_to_config[trial_id]
            if self.max_resource_attr is not None:
                config = dict(_config, **{self.max_resource_attr: slot_in_rung.level})
            else:
                config = _config
            suggestion = TrialSuggestion.resume_suggestion(
                trial_id=trial_id, config=config
            )
            if do_debug_log:
                logger.info(f"trial_id {trial_id} promoted to {slot_in_rung.level}")
        else:
            # New trial to be started (id is ``trial_id`` passed in)
            config = self.searcher.get_config(trial_id=str(trial_id))
            if config is not None:
                config = cast_config_values(config, self.config_space)
                self.searcher.register_pending(
                    trial_id=str(trial_id), config=config, milestone=slot_in_rung.level
                )
                if self.max_resource_attr is not None:
                    config[self.max_resource_attr] = slot_in_rung.level
                self._trial_to_config[trial_id] = config
                suggestion = TrialSuggestion.start_suggestion(config=config)
                # Assign trial id to job descriptor
                slot_in_rung.trial_id = trial_id
                if do_debug_log:
                    logger.info(
                        f"trial_id {trial_id} starts (milestone = "
                        f"{slot_in_rung.level})"
                    )
        if suggestion is not None:
            assert trial_id not in self._trial_to_pending_slot, (
                f"Trial for trial_id = {trial_id} is already registered as "
                + "pending, cannot resume or start it"
            )
            self._trial_to_pending_slot[trial_id] = (bracket_id, slot_in_rung)
        else:
            # Searcher failed to return a config for a new trial_id. We report
            # the corresponding job as failed, so that in case the experiment
            # is continued, the bracket is not blocked with a slot which remains
            # pending forever
            logger.warning(
                "Searcher failed to suggest a configuration for new trial "
                f"{trial_id}. The corresponding rung slot is marked as failed."
            )
            self._report_as_failed(bracket_id, slot_in_rung)
        return suggestion

    def _on_result(self, result: Tuple[int, SlotInRung]):
        trials_not_promoted = self.bracket_manager.on_result(result)
        if trials_not_promoted is not None:
            self._trials_checkpoints_can_be_removed.extend(trials_not_promoted)

    def _report_as_failed(self, bracket_id: int, slot_in_rung: SlotInRung):
        result_failed = SlotInRung(
            rung_index=slot_in_rung.rung_index,
            level=slot_in_rung.level,
            slot_index=slot_in_rung.slot_index,
            trial_id=slot_in_rung.trial_id,
            metric_val=np.NAN,
        )
        self._on_result((bracket_id, result_failed))

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        trial_id = trial.trial_id
        if trial_id in self._trial_to_pending_slot:
            bracket_id, slot_in_rung = self._trial_to_pending_slot[trial_id]
            assert slot_in_rung.trial_id == trial_id  # Sanity check
            assert self.metric in result, (
                f"Result for trial_id {trial_id} does not contain "
                + f"'{self.metric}' field"
            )
            metric_val = float(result[self.metric])
            assert self._resource_attr in result, (
                f"Result for trial_id {trial_id} does not contain "
                + f"'{self._resource_attr}' field"
            )
            resource = int(result[self._resource_attr])
            milestone = slot_in_rung.level
            trial_decision = SchedulerDecision.CONTINUE
            if resource >= milestone:
                assert resource == milestone, (
                    f"Trial trial_id {trial_id}: Obtained result for "
                    + f"resource = {resource}, but not for {milestone}. "
                    + "Training script must not skip rung levels!"
                )
                # Reached rung level: Pass result to bracket manager
                slot_in_rung.metric_val = metric_val
                self._on_result((bracket_id, slot_in_rung))
                # Remove it from pending slots
                del self._trial_to_pending_slot[trial_id]
                # Trial should be paused
                trial_decision = SchedulerDecision.PAUSE
            prev_level = self.bracket_manager.level_to_prev_level(bracket_id, milestone)
            if resource > prev_level:
                # If the training script does not implement checkpointing, each
                # trial starts from scratch. In this case, the condition
                # ``resource > prev_level`` ensures that the searcher does not
                # receive multiple reports for the same resource
                update = self.searcher_data == "all" or resource == milestone
                self.searcher.on_trial_result(
                    trial_id=str(trial_id),
                    config=self._trial_to_config[trial_id],
                    result=result,
                    update=update,
                )
        else:
            trial_decision = SchedulerDecision.STOP
            logger.warning(
                f"Received result for trial_id {trial_id}, which is not "
                f"pending. This result is not used:\n{result}"
            )

        return trial_decision

    def on_trial_error(self, trial: Trial):
        """
        Given the ``trial`` is currently pending, we send a result at its
        milestone for metric value NaN. Such trials are ranked after all others
        and will most likely not be promoted.

        """
        super().on_trial_error(trial)
        trial_id = trial.trial_id
        if trial_id in self._trial_to_pending_slot:
            bracket_id, slot_in_rung = self._trial_to_pending_slot[trial_id]
            self._report_as_failed(bracket_id, slot_in_rung)
            # A failed trial is not pending anymore
            del self._trial_to_pending_slot[trial_id]
        else:
            logger.warning(
                f"Trial trial_id {trial_id} not registered at pending: "
                "on_trial_error call is ignored"
            )

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return self.mode

    def trials_checkpoints_can_be_removed(self) -> List[int]:
        result = self._trials_checkpoints_can_be_removed
        self._trials_checkpoints_can_be_removed = []
        return result
