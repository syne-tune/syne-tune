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
import logging
from typing import Dict, Optional
import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision, TrialSuggestion
from syne_tune.config_space import cast_config_values
from syne_tune.backend.time_keeper import RealTimeKeeper
from syne_tune.optimizer.schedulers.neuralbands.neuralband_supplement import (
    NeuralbandSchedulerBase,
)


logger = logging.getLogger(__name__)


def is_continue_decision(trial_decision: str) -> bool:
    return trial_decision == SchedulerDecision.CONTINUE


class NeuralbandScheduler(NeuralbandSchedulerBase):
    def __init__(
        self,
        config_space: Dict,
        gamma: float = 0.01,
        nu: float = 0.01,
        step_size: int = 30,
        max_while_loop: int = 100,
        **kwargs,
    ):
        """
        NeuralBand is a neural-bandit based HPO algorithm for the multi-fidelity setting. It uses a budget-aware neural
        network together with a feedback perturbation to efficiently explore the input space across fidelities.
        NeuralBand uses a novel configuration selection criterion to actively choose the configuration in each trial
        and incrementally exploits the knowledge of every past trial.

        :param config_space:
        :param gamma: Control aggressiveness of configuration selection criterion
        :param nu: Control aggressiveness of perturbing feedback for exploration
        :param step_size: How many trials we train network once
        :param max_while_loop: Maximal number of times we can draw a configuration from configuration space
        :param kwargs:
        """
        super(NeuralbandScheduler, self).__init__(
            config_space, step_size, max_while_loop, **kwargs
        )
        self.gamma = gamma
        self.nu = nu

        if self.mode == "min":
            self.max_while_loop = max_while_loop
        else:
            self.max_while_loop = 2

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        self._initialize_searcher()
        # If no time keeper was provided at construction, we use a local
        # one which is started here
        if self.time_keeper is None:
            self.time_keeper = RealTimeKeeper()
            self.time_keeper.start_of_time()
        # For pause/resume schedulers: Can a paused trial be promoted?
        promote_trial_id, extra_kwargs = self._promote_trial()
        if promote_trial_id is not None:
            promote_trial_id = int(promote_trial_id)
            return TrialSuggestion.resume_suggestion(
                trial_id=promote_trial_id, config=extra_kwargs
            )
        # Ask searcher for config of new trial to start
        extra_kwargs["elapsed_time"] = self._elapsed_time()
        trial_id = str(trial_id)

        # active selection criterion
        initial_budget = self.net.max_b
        while_loop_count = 0
        l_t_score = []
        while 1:
            config = self.searcher.get_config(**extra_kwargs, trial_id=trial_id)
            if config is not None:
                config_encoding = self.hp_ranges.to_ndarray(config)
                predict_score = self.net.predict(
                    (config_encoding, initial_budget)
                ).item()
                l_t_score.append((config, predict_score))
                if self.mode == "min":
                    if (
                        self.currnet_best_score - predict_score
                        > self.gamma
                        * self.currnet_best_score
                        * (1.0 - initial_budget / self.max_t)
                    ):
                        break
                    if while_loop_count > self.max_while_loop:
                        l_t_score = sorted(l_t_score, key=lambda x: x[1])
                        config = l_t_score[0][0]
                        break
                else:
                    if predict_score * 100.0 - self.currnet_best_score > self.gamma * (
                        100.0 - self.currnet_best_score
                    ) * (1.0 - initial_budget / self.max_t):
                        break
                    if while_loop_count > self.max_while_loop:
                        l_t_score = sorted(l_t_score, key=lambda x: x[1], reverse=True)
                        config = l_t_score[0][0]
                        break
                while_loop_count += 1
            else:
                self._searcher_initialized = False
                self._initialize_searcher_new()
                config = self.searcher.get_config(**extra_kwargs, trial_id=trial_id)
                break

        if config is not None:
            config = cast_config_values(config, self.config_space)
            config = self._on_config_suggest(config, trial_id, **extra_kwargs)
            config = TrialSuggestion.start_suggestion(config)

        return config

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
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
            if trial_id not in self._active_trials:
                # Trial not in self._active_trials anymore, so must have been
                # stopped
                trial_decision = SchedulerDecision.STOP
                logger.warning(
                    f"trial_id {trial_id}: Was STOPPED, but receives another "
                    f"report {result}\nThis report is ignored"
                )
            elif not self._active_trials[trial_id]["running"]:
                # Trial must have been paused before
                trial_decision = SchedulerDecision.PAUSE
                logger.warning(
                    f"trial_id {trial_id}: Was PAUSED, but receives another "
                    f"report {result}\nThis report is ignored"
                )
            else:
                # perturb the feedback and train network
                config = trial.config
                config_encoding = self.hp_ranges.to_ndarray(config)
                if "epoch" in result:
                    hp_budget = float(result["epoch"] / self.max_t)
                else:
                    hp_budget = float(result["hp_epoch"] / self.max_t)
                test_loss = result[self.metric]
                # update current best score
                if self.mode == "min":
                    if test_loss < self.currnet_best_score:
                        self.currnet_best_score = test_loss
                    perturbed_loss = test_loss + np.random.normal(
                        0, self.nu * self.currnet_best_score * (1 - hp_budget)
                    )
                else:
                    if test_loss > self.currnet_best_score:
                        self.currnet_best_score = test_loss
                    perturbed_loss = (
                        test_loss
                        + np.random.normal(
                            0, self.nu * (100.0 - test_loss) * (1 - hp_budget)
                        )
                    ) / 100.0
                self.net.add_data((config_encoding, hp_budget), perturbed_loss)

                # train network
                if self.net.data_size % self.train_step_size == 0:
                    predict_score = self.net.predict(
                        (config_encoding, hp_budget)
                    ).item()
                    self.net.train()

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
                        if self._cost_offset[trial_id] > 0:
                            logger.info(
                                f"trial_id {trial_id}: Resumed trial seems to have been "
                                + "started from scratch (no checkpointing?), so we erase "
                                + "the cost offset."
                            )
                        self._cost_offset[trial_id] = 0

                do_update = self._update_searcher(trial_id, config, result, task_info)
                resource = int(result[self._resource_attr])
                self._active_trials[trial_id].update(
                    {
                        "time_stamp": time_since_start,
                        "reported_result": {
                            self.metric: result[self.metric],
                            self._resource_attr: resource,
                        },
                        "keep_case": milestone_reached,
                    }
                )
                if do_update:
                    largest_update_resource = self._active_trials[trial_id][
                        "largest_update_resource"
                    ]
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
                        self._active_trials[trial_id][
                            "largest_update_resource"
                        ] = resource
                if not task_continues:
                    if (not self.does_pause_resume()) or resource >= self.max_t:
                        trial_decision = SchedulerDecision.STOP
                        act_str = "Terminating"
                    else:
                        trial_decision = SchedulerDecision.PAUSE
                        act_str = "Pausing"
                    self._cleanup_trial(trial_id)
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
