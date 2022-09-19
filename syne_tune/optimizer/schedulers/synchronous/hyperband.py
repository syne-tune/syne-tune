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
from typing import Optional, List
import logging
import numpy as np

from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket_manager import (
    SynchronousHyperbandBracketManager,
)
from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket import SlotInRung
from syne_tune.optimizer.schedulers.synchronous.hyperband_rung_system import (
    RungSystemsPerBracket,
)
from syne_tune.optimizer.scheduler import TrialSuggestion, SchedulerDecision
from syne_tune.optimizer.schedulers.fifo import ResourceLevelsScheduler
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import cast_config_values
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    Categorical,
    String,
    assert_no_invalid_options,
    Integer,
)
from syne_tune.optimizer.schedulers.random_seeds import RandomSeedGenerator
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_factory

__all__ = ["SynchronousHyperbandScheduler"]

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    "searcher",
    "search_options",
    "metric",
    "mode",
    "points_to_evaluate",
    "random_seed",
    "max_resource_attr",
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
    "random_seed": Integer(0, 2**32 - 1),
    "max_resource_attr": String(),
    "resource_attr": String(),
    "searcher_data": Categorical(("rungs", "all")),
}


class SynchronousHyperbandScheduler(ResourceLevelsScheduler):
    """
    Synchronous Hyperband. This is still scheduling jobs asynchronously, but
    decision-making is synchronized, in that trials are only promoted to the
    next milestone once the rung they are currently paused at, is completely
    occupied.

    Our implementation never delays scheduling of a job. If the currently
    active bracket does not accept jobs, we assign the job to a later bracket.
    This means that at any point in time, several brackets can be active, but
    jobs are preferentially assigned to the first one (the "primary" active
    bracket).

    Parameters
    ----------
    config_space : dict
        Configuration space for trial evaluation function
    bracket_rungs : RungSystemsPerBracket
        Determines rung level systems for each bracket, see
        :class:`SynchronousHyperbandBracketManager`
    searcher : str
        Selects searcher. Passed to `searcher_factory`.
        NOTE: Different to :class:`FIFOScheduler`, we do not accept a
        `BaseSearcher` object here.
    search_options : dict
        Passed to `searcher_factory`
    metric : str
        Name of metric to optimize, key in result's obtained via
        `on_trial_result`
    mode : str
        Mode to use for the metric given, can be 'min' or 'max'
    points_to_evaluate: list[dict] or None
        List of configurations to be evaluated initially (in that order).
        Each config in the list can be partially specified, or even be an
        empty dict. For each hyperparameter not specified, the default value
        is determined using a midpoint heuristic.
        If None (default), this is mapped to [dict()], a single default config
        determined by the midpoint heuristic. If [] (empty list), no initial
        configurations are specified.
        Note: If `searcher` is BaseSearcher, points_to_evaluate must be set
        there.
    random_seed : int
        Master random seed. Generators used in the scheduler or searcher are
        seeded using `RandomSeedGenerator`. If not given, the master random
        seed is drawn at random here.
    max_resource_attr : str
        Key name in config for fixed attribute containing the maximum resource.
        If given, trials need not be stopped, which can run more efficiently.
    resource_attr : str
        Name of resource attribute in result's obtained via `on_trial_result`.
        Note: The type of resource must be int.
    searcher_data : str
        Relevant only if a model-based searcher is used.
        Example: For NN tuning and `resource_attr == epoch', we receive a
        result for each epoch, but not all epoch values are also rung levels.
        searcher_data determines which of these results are passed to the
        searcher. As a rule, the more data the searcher receives, the better
        its fit, but also the more expensive get_config may become. Choices:
        - 'rungs' (default): Only results at rung levels. Cheapest
        - 'all': All results. Most expensive
        Note: For a Gaussian additive learning curve surrogate model, this
        has to be set to 'all'.

    """

    def __init__(
        self, config_space: dict, bracket_rungs: RungSystemsPerBracket, **kwargs
    ):
        super().__init__(config_space)
        self._create_internal(bracket_rungs, **kwargs)

    def _create_internal(self, bracket_rungs: RungSystemsPerBracket, **kwargs):
        # Check values and impute default values
        assert_no_invalid_options(
            kwargs, _ARGUMENT_KEYS, name="SynchronousHyperbandScheduler"
        )
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
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
        # Generator for random seeds
        random_seed = kwargs.get("random_seed")
        if random_seed is None:
            random_seed = np.random.randint(0, 2**32)
        logger.info(f"Master random_seed = {random_seed}")
        self.random_seed_generator = RandomSeedGenerator(random_seed)
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
            # We need `max_epochs` in this case
            max_epochs = self._infer_max_resource_level(
                max_resource_level=None, max_resource_attr=self.max_resource_attr
            )
            assert max_epochs is not None, (
                "If searcher='bayesopt', need to know the maximum resource "
                + "level. Please provide max_resource_attr argument."
            )
            search_options["max_epochs"] = max_epochs
        self.searcher: BaseSearcher = searcher_factory(searcher, **search_options)
        # Bracket manager
        self.bracket_manager = SynchronousHyperbandBracketManager(
            bracket_rungs, mode=self.mode
        )
        self.searcher_data = kwargs["searcher_data"]
        # Maps trial_id to tuples (bracket_id, slot_in_rung), as returned
        # by `bracket_manager.next_job`, and required by
        # `bracket_manager.on_result`. Entries are removed once passed to
        # `on_result`. Note that a trial_id can be associated with different
        # job descriptions in its lifetime
        self._trial_to_pending_slot = dict()
        # Maps trial_id (active) to config
        self._trial_to_config = dict()

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        do_debug_log = self.searcher.debug_log is not None
        if do_debug_log and trial_id == 0:
            # This is printed at the start of the experiment. Cannot do this
            # at construction, because with `RemoteLauncher` this does not end
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
            # Paused trial to be resumed (`trial_id` passed in is ignored)
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
            # New trial to be started (id is `trial_id` passed in)
            config = self.searcher.get_config(trial_id=str(trial_id))
            if config is not None:
                config = cast_config_values(config, self.config_space)
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

    def _report_as_failed(self, bracket_id: int, slot_in_rung: SlotInRung):
        result_failed = SlotInRung(
            rung_index=slot_in_rung.rung_index,
            level=slot_in_rung.level,
            slot_index=slot_in_rung.slot_index,
            trial_id=slot_in_rung.trial_id,
            metric_val=np.NAN,
        )
        self.bracket_manager.on_result((bracket_id, result_failed))

    def on_trial_result(self, trial: Trial, result: dict) -> str:
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
            prev_level = self.bracket_manager.level_to_prev_level(bracket_id, milestone)
            trial_decision = SchedulerDecision.CONTINUE
            if resource >= milestone:
                assert resource == milestone, (
                    f"Trial trial_id {trial_id}: Obtained result for "
                    + f"resource = {resource}, but not for {milestone}. "
                    + "Training script must not skip rung levels!"
                )
                # Reached rung level: Pass result to bracket manager
                slot_in_rung.metric_val = metric_val
                self.bracket_manager.on_result((bracket_id, slot_in_rung))
                # Remove it from pending slots
                del self._trial_to_pending_slot[trial_id]
                # Trial should be paused
                trial_decision = SchedulerDecision.PAUSE
            if resource > prev_level:
                # If the training script does not implement checkpointing, each
                # trial starts from scratch. In this case, the condition
                # `resource > prev_level` ensures that the searcher does not
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
        Given the `trial` is currently pending, we send a result at its
        milestone for metric value NaN. Such trials are ranked after all others
        and will most likely not be promoted.

        """
        trial_id = trial.trial_id
        self.searcher.evaluation_failed(str(trial_id))
        if trial_id in self._trial_to_pending_slot:
            bracket_id, slot_in_rung = self._trial_to_pending_slot[trial_id]
            self._report_as_failed(bracket_id, slot_in_rung)
        else:
            logger.warning(
                f"Trial trial_id {trial_id} not registered at pending: "
                "on_trial_error call is ignored"
            )

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return self.mode
