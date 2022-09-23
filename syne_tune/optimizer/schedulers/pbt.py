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
import numpy as np

from dataclasses import dataclass
from collections import deque
from typing import Callable, List, Optional, Tuple

from syne_tune.config_space import Domain, Integer, Float, FiniteRange
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision, TrialSuggestion
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.config_space import cast_config_values
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    Integer as DA_Integer,
    Boolean as DA_Boolean,
    filter_by_key,
    String as DA_String,
    Float as DA_Float,
)

logger = logging.getLogger(__name__)


@dataclass
class PBTTrialState:
    """Internal PBT state tracked per-trial."""

    trial: Trial
    last_score: float = None
    last_checkpoint: int = None
    last_perturbation_time: int = 0
    stopped: bool = False


_ARGUMENT_KEYS = {
    "resource_attr",
    "population_size",
    "perturbation_interval",
    "quantile_fraction",
    "resample_probability",
    "log_config",
}

_DEFAULT_OPTIONS = {
    "resource_attr": "time_total_s",
    "population_size": 4,
    "perturbation_interval": 60.0,
    "quantile_fraction": 0.25,
    "resample_probability": 0.25,
    "log_config": True,
}

_CONSTRAINTS = {
    "resource_attr": DA_String(),
    "population_size": DA_Integer(1, None),
    "perturbation_interval": DA_Float(0.01, None),
    "quantile_fraction": DA_Float(0.0, 0.5),
    "resample_probability": DA_Float(0.0, 1.0),
    "log_config": DA_Boolean(),
}


class PopulationBasedTraining(FIFOScheduler):
    """
    Implements the Population Based Training (PBT) algorithm. This is an adapted version of
    the Raytune implementation for Syne Tune:
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

    The mutation happens as following. For each hyperparameter, we either resample
    its value uniformly at random, or otherwise increment (multiply by 1.2) or
    decrement (multiply by 0.8) the value (probability 0.5 each). For categorical
    hyperparameters, the value is always resampled uniformly.

    Note: While this is implemented as child of :class:`FIFOScheduler`, we
    require `searcher='random'` (default), since the current code only supports
    a random searcher.

    Parameters
    ----------
    config_space: dict
        Configuration space for trial evaluation function
    search_options : dict
        If searcher is str, these arguments are passed to searcher_factory.
    metric : str
        Name of metric to optimize, key in result's obtained via
        `on_trial_result`
    mode : str
        Mode to use for the metric given, can be 'min' or 'max', default to 'min'.
    points_to_evaluate: list[dict] or None
        See :class:`FIFOScheduler`
    random_seed : int
        Master random seed. Generators used in the scheduler or searcher are
        seeded using `RandomSeedGenerator`. If not given, the master random
        seed is drawn at random here.
    max_t : int (optional)
        See :class:`FIFOScheduler`. This is mandatory here. If not given, we
        try to infer it.
    max_resource_attr : str (optional)
        See :class:`FIFOScheduler`.
    population_size : int)
        Defines the size of the population.
    perturbation_interval : float
        Models will be considered for perturbation at this interval of `time_attr`.
        Note that perturbation incurs checkpoint overhead, so you shouldn't set
        this to be too frequent.
    quantile_fraction : float
        Parameters are transferred from the top `quantile_fraction` fraction of
        trials to the bottom `quantile_fraction` fraction. Needs to be between
        0 and 0.5. Setting it to 0 essentially implies doing no exploitation at
        all.
    resample_probability : float
        The probability of resampling from the original distribution when
        applying `hyperparam_mutations`. If not resampled, the value will be
        perturbed by a factor of 1.2 or 0.8 if continuous, or changed to an
        adjacent value if discrete.
    custom_explore_fn : callable
        You can also specify a custom exploration function. This function is
        invoked as `f(config)` after built-in perturbations from
        `hyperparam_mutations` are applied, and should return `config` updated
        as needed. You must specify at least one of `hyperparam_mutations` or
        `custom_explore_fn`.
    log_config (bool): Whether to log the ray config of each model to\
        `local_dir` at each exploit. Allows config schedule to be reconstructed.
    time_keeper : TimeKeeper
        See :class:`FIFOScheduler`
    """

    def __init__(
        self,
        config_space: dict,
        custom_explore_fn: Optional[Callable[[dict], dict]] = None,
        **kwargs,
    ):
        # The current implementation only supports a random searcher
        searcher = kwargs.get("searcher")
        if searcher is not None:
            assert (
                isinstance(searcher, str) and searcher == "random"
            ), "PopulationBasedTraining only supports searcher='random' for now"
        kwargs = check_and_merge_defaults(
            kwargs, set(), _DEFAULT_OPTIONS, _CONSTRAINTS, dict_name="scheduler_options"
        )
        self._resource_attr = kwargs["resource_attr"]
        self._population_size = kwargs["population_size"]
        self._perturbation_interval = kwargs["perturbation_interval"]
        self._quantile_fraction = kwargs["quantile_fraction"]
        self._resample_probability = kwargs["resample_probability"]
        self._custom_explore_fn = custom_explore_fn
        self._log_config = kwargs["log_config"]
        # Superclass constructor
        super().__init__(config_space, **filter_by_key(kwargs, _ARGUMENT_KEYS))
        assert self.max_t is not None, (
            "Either max_t must be specified, or it has to be specified as "
            + "config_space['epochs'], config_space['max_t'], "
            + "config_space['max_epochs']"
        )

        self._metric_op = 1.0 if self.mode == "max" else -1.0
        self._trial_state = {}
        self._next_perturbation_sync = self._perturbation_interval
        self._trial_decisions_stack = deque()
        self._checkpointing_history = []
        self._num_checkpoints = 0
        self._num_perturbations = 0
        self._random_state = np.random.RandomState(self.random_seed_generator())

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
            trial_id_to_clone = self._random_state.choice(upper_quantile)
            assert trial.trial_id is not trial_id_to_clone
            return trial_id_to_clone
        else:
            return trial.trial_id

    def on_trial_result(self, trial: Trial, result: dict) -> str:
        if self._resource_attr not in result:
            time_missing_msg = (
                f"Cannot find resource_attr {self._resource_attr} "
                f"in trial result {result}. Make sure that this "
                "attribute is returned in the "
                "results of your Trainable."
            )
            raise RuntimeError(time_missing_msg)
        if self.metric not in result:
            metric_missing_msg = (
                f"Cannot find metric {self.metric} in trial result {result}. "
                "Make sure that this attribute is returned "
                "in the "
                "results of your Trainable."
            )
            raise RuntimeError(metric_missing_msg)

        cost = result[self._resource_attr]
        state = self._trial_state[trial.trial_id]

        # Stop if we reached the maximum budget of this configuration
        if cost >= self.max_t:
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
            (trial.trial_id, trial_id_to_continue, self._elapsed_time())
        )

        if trial_id_to_continue == trial.trial_id:
            # continue current trial
            return SchedulerDecision.CONTINUE
        else:
            state.stopped = True
            # exploit step
            trial_to_clone = self._trial_state[trial_id_to_continue].trial

            # explore step
            config = self._explore(trial_to_clone.config)
            self._trial_decisions_stack.append((trial_id_to_continue, config))
            return SchedulerDecision.PAUSE

    def _save_trial_state(self, state: PBTTrialState, time: int, result: dict) -> dict:
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
                math.ceil(len(trials) * self._quantile_fraction)
            )
            if num_trials_in_quantile > len(trials) / 2:
                num_trials_in_quantile = int(math.floor(len(trials) / 2))
            return (trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:])

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        # If no time keeper was provided at construction, we use a local
        # one which is started here
        if len(self._trial_decisions_stack) == 0:
            # If our stack is empty, we simply start a new random configuration.
            return super()._suggest(trial_id)
        else:
            assert self.time_keeper is not None  # Sanity check
            trial_id_to_continue, config = self._trial_decisions_stack.pop()
            config["elapsed_time"] = self._elapsed_time()
            config = cast_config_values(
                config=config, config_space=self.searcher.config_space
            )
            config["trial_id"] = trial_id
            return TrialSuggestion.start_suggestion(
                config=config, checkpoint_trial_id=trial_id_to_continue
            )

    def _explore(self, config: dict) -> dict:
        """Return a config perturbed as specified.

        Args:
            config (dict): Original hyperparameter configuration from the cloned trial
        """

        new_config = copy.deepcopy(config)

        self._num_perturbations += 1

        if self._custom_explore_fn:
            new_config = self._custom_explore_fn(new_config)
            assert (
                new_config is not None
            ), "Custom explore fn failed to return new config"
            return new_config

        for key, hp_range in self.config_space.items():
            if isinstance(hp_range, Domain):
                # For `Categorical`, all values have the same distance from each
                # other, so we can always resample uniformly
                is_numerical = (
                    isinstance(hp_range, Float)
                    or isinstance(hp_range, Integer)
                    or isinstance(hp_range, FiniteRange)
                )
                if (
                    not is_numerical
                ) or self._random_state.rand() < self._resample_probability:
                    new_config[key] = hp_range.sample(
                        size=1, random_state=self._random_state
                    )
                else:
                    multiplier = 1.2 if self._random_state.rand() > 0.5 else 0.8
                    new_config[key] = hp_range.cast(
                        np.clip(
                            config[key] * multiplier, hp_range.lower, hp_range.upper
                        )
                    )

        # Only log mutated hyperparameters and not entire config.
        old_hparams = {k: v for k, v in config.items() if k in self.config_space}
        new_hparams = {k: v for k, v in new_config.items() if k in self.config_space}
        logger.debug(f"[explore] perturbed config from {old_hparams} -> {new_hparams}")

        return new_config
