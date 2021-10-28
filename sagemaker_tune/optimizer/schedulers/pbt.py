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
import math
import random
import numpy as np

from dataclasses import dataclass
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

from sagemaker_tune.search_space import Domain
from sagemaker_tune.backend.trial_status import Trial
from sagemaker_tune.optimizer.scheduler import SchedulerDecision, TrialSuggestion
from sagemaker_tune.optimizer.schedulers.fifo import FIFOScheduler
from sagemaker_tune.search_space import cast_config_values
from sagemaker_tune.backend.time_keeper import RealTimeKeeper

logger = logging.getLogger(__name__)


@dataclass
class PBTTrialState:
    """Internal PBT state tracked per-trial."""

    trial: Trial
    last_score: float = None
    last_checkpoint: int = None
    last_perturbation_time: int = 0
    stopped: bool = False


class PopulationBasedTraining(FIFOScheduler):
    """
    Implements the Population Based Training (PBT) algorithm. This is an adapted version of
    the Raytune implementation for SagemakerTune:
     https://docs.ray.io/en/latest/tune/tutorials/tune-advanced-tutorial.html

    PBT was original presented in the following paper:
    https://deepmind.com/blog/population-based-training-neural-networks


    Population based training (PBT) maintains a population of neural network models spread across
    an asynchronous set of workers and dynamically adjust their hyperparameters during training.
    Every time a worker reaches a user-defined milestone, it returns the performance of the currently
    evaluated network. If the network is within the top percentile of the population,
    the worker resumes its training until the next milestone. If not, PBT selects a neural network
    from the top percentile uniformly at random. The worker now continues with the latest checkpoint
    of this new neural network but mutates the hyperparameters.
    The mutation happens as following: With a some probability a new set of hyperparameters
    is sampled uniformly at random. Otherwise a subsets of the current hyperparameters is either
    increment (multiplied by 1.2) or decremented (multiplied by 0.8).


    Args:
        config_space (dict): Configuration space for trial evaluation function
        metric (str): The training result objective value attribute. Stopping
            procedures will use this attribute.
        mode (str): One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        resource_attr (str): The resource attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        population_size (int): Defines the size of the population.
        max_t (int): Maximum resource (see resource_attr) to be used for a job.
        perturbation_interval (float): Models will be considered for
            perturbation at this interval of `time_attr`. Note that
            perturbation incurs checkpoint overhead, so you shouldn't set this
            to be too frequent.
        quantile_fraction (float): Parameters are transferred from the top
            `quantile_fraction` fraction of trials to the bottom
            `quantile_fraction` fraction. Needs to be between 0 and 0.5.
            Setting it to 0 essentially implies doing no exploitation at all.
        resample_probability (float): The probability of resampling from the
            original distribution when applying `hyperparam_mutations`. If not
            resampled, the value will be perturbed by a factor of 1.2 or 0.8
            if continuous, or changed to an adjacent value if discrete.
        custom_explore_fn (func): You can also specify a custom exploration
            function. This function is invoked as `f(config)` after built-in
            perturbations from `hyperparam_mutations` are applied, and should
            return `config` updated as needed. You must specify at least one of
            `hyperparam_mutations` or `custom_explore_fn`.
        log_config (bool): Whether to log the ray config of each model to
            local_dir at each exploit. Allows config schedule to be
            reconstructed.
        time_keeper: See :class:`FIFOScheduler`
    """

    def __init__(self,
                 config_space: Dict,
                 metric: str,
                 max_t: int,
                 mode: str = 'min',
                 resource_attr: str = "time_total_s",
                 population_size: int = 4,
                 perturbation_interval: float = 60.0,
                 quantile_fraction: float = 0.25,
                 resample_probability: float = 0.25,
                 custom_explore_fn: Optional[Callable[[Dict], Dict]] = None,
                 log_config: bool = True,
                 **kwargs,
                 ):

        super().__init__(config_space, metric=metric, mode=mode, **kwargs)

        if quantile_fraction > 0.5 or quantile_fraction < 0:
            raise ValueError(
                "You must set `quantile_fraction` to a value between 0 and"
                "0.5. Current value: '{}'".format(quantile_fraction))

        if perturbation_interval <= 0:
            raise ValueError(
                "perturbation_interval must be a positive number greater "
                "than 0. Current value: '{}'".format(perturbation_interval))

        assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."

        self._metric_op = None
        if self.mode == "max":
            self._metric_op = 1.
        elif self.mode == "min":
            self._metric_op = -1.

        self._resource_attr = resource_attr
        self._perturbation_interval = perturbation_interval
        self._quantile_fraction = quantile_fraction
        self._resample_probability = resample_probability
        self._trial_state = {}
        self._custom_explore_fn = custom_explore_fn
        self._log_config = log_config
        self._next_perturbation_sync = self._perturbation_interval
        self._trial_decisions_stack = deque()
        self._population_size = population_size
        self._checkpointing_history = []
        self._max_t = max_t
        self._num_checkpoints = 0
        self._num_perturbations = 0

    def on_trial_add(self, trial: Trial):
        self._trial_state[trial.trial_id] = PBTTrialState(trial=trial)

    def _get_trial_id_to_continue(self, trial: Trial):
        """
        Determine which trial to continue. Following the original PBT formulation if the trial is not in the top %n
        percent, we sample a trial uniformly at random from the upper quantile.
        :param trial:
        :return: int that specifies which trial should be continued
        """
        lower_quantile, upper_quantile = self._quantiles()
        # If we are not in the upper quantile, we pause:
        if trial.trial_id in lower_quantile:
            logger.debug(f"Trial {trial.trial_id} is in lower quantile")
            # sample random trial from upper quantile
            trial_id_to_clone = random.choice(upper_quantile)
            assert trial.trial_id is not trial_id_to_clone
            return trial_id_to_clone
        else:
            return trial.trial_id

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
        if self._resource_attr not in result:
            time_missing_msg = f"Cannot find resource_attr {self._resource_attr} " \
                               f"in trial result {result}. Make sure that this " \
                               "attribute is returned in the " \
                               "results of your Trainable."
            raise RuntimeError(time_missing_msg)
        if self.metric not in result:
            metric_missing_msg = f"Cannot find metric {self.metric} in trial result {result}. " \
                                 "Make sure that this attribute is returned " \
                                 "in the " \
                                 "results of your Trainable."
            raise RuntimeError(metric_missing_msg)

        cost = result[self._resource_attr]
        state = self._trial_state[trial.trial_id]

        # Stop if we reached the maximum budget of this configuration
        if cost >= self._max_t:
            state.stopped = True
            return SchedulerDecision.STOP

        # Continue training if perturbation interval has not been reached yet.
        if cost - state.last_perturbation_time < self._perturbation_interval:
            return SchedulerDecision.CONTINUE

        self._save_trial_state(state, cost, result)

        state.last_perturbation_time = cost

        trial_id_to_continue = self._get_trial_id_to_continue(trial)

        # bookkeeping for debugging reasons
        self._checkpointing_history.append(
            (trial.trial_id, trial_id_to_continue, self._elapsed_time()))

        if trial_id_to_continue == trial.trial_id:
            # continue current trial
            return SchedulerDecision.CONTINUE
        else:
            state.stopped = True
            # exploit step
            trial_to_clone = self._trial_state[trial_id_to_continue].trial

            # explore step
            config = self._explore(trial_to_clone.config)
            self._trial_decisions_stack.append((trial_id_to_continue,  config))
            return SchedulerDecision.PAUSE

    def _save_trial_state(self, state: PBTTrialState, time: int, result: Dict) -> Dict:
        """Saves necessary trial information when result is received.
        Args:
            state (PBTTrialState): The state object for the trial.
            time (int): The current timestep of the trial.
            result (dict): The trial's result dictionary.
        """

        # This trial has reached its perturbation interval.
        # Record new state in the state object.
        score = self._metric_op * result[self.metric]
        state.last_score = score
        state.last_train_time = time
        state.last_result = result

        return score

    def _quantiles(self) -> Tuple[List[Trial], List[Trial]]:
        """Returns trials in the lower and upper `quantile` of the population.

        If there is not enough data to compute this, returns empty lists.
        """
        trials = []
        for trial, state in self._trial_state.items():
            if not state.stopped and state.last_score is not None:
                trials.append(trial)

        trials.sort(key=lambda t: self._trial_state[t].last_score)

        if len(trials) <= 1:
            return [], []
        else:
            num_trials_in_quantile = int(
                math.ceil(len(trials) * self._quantile_fraction))
            if num_trials_in_quantile > len(trials) / 2:
                num_trials_in_quantile = int(math.floor(len(trials) / 2))
            return (trials[:num_trials_in_quantile],
                    trials[-num_trials_in_quantile:])

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        # If no time keeper was provided at construction, we use a local
        # one which is started here
        if self.time_keeper is None:
            self.time_keeper = RealTimeKeeper()
            self.time_keeper.start_of_time()
        if len(self._trial_decisions_stack) == 0:
            # If our stack is empty, we simply start a new random configuration.
            return super()._suggest(trial_id)
        else:
            trial_id_to_continue, config = self._trial_decisions_stack.pop()
            config['elapsed_time'] = self._elapsed_time()
            config = cast_config_values(
                config=config,
                config_space=self.searcher.configspace)
            config['trial_id'] = trial_id
            return TrialSuggestion.start_suggestion(config=config,
                                                    checkpoint_trial_id=trial_id_to_continue)

    def _explore(self, config: Dict) -> Dict:
        """Return a config perturbed as specified.

        Args:
            config (dict): Original hyperparameter configuration from the cloned trial
        """

        new_config = copy.deepcopy(config)

        self._num_perturbations += 1

        if self._custom_explore_fn:
            new_config = self._custom_explore_fn(new_config)
            assert new_config is not None, \
                "Custom explore fn failed to return new config"
            return new_config

        for key, distribution in self.config_space.items():
            if not isinstance(distribution, Domain):
                continue
            else:
                if random.random() < self._resample_probability:
                    new_config[key] = distribution.sample(size=1) if isinstance(
                        distribution, Domain) else distribution()
                elif random.random() > 0.5:
                    new_config[key] = np.clip(config[key] * 1.2, distribution.lower, distribution.upper)
                else:
                    new_config[key] = np.clip(config[key] * 0.8, distribution.lower, distribution.upper)
                if isinstance(config[key], int):
                    new_config[key] = int(new_config[key])

        # Only log mutated hyperparameters and not entire config.
        old_hparams = {
            k: v
            for k, v in config.items()
            if k in self.config_space
        }
        new_hparams = {
            k: v
            for k, v in new_config.items() if k in self.config_space
        }
        logger.debug(f"[explore] perturbed config from {old_hparams} -> {new_hparams}")

        return new_config
