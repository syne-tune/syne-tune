import copy
import logging
import math
import numpy as np

from dataclasses import dataclass
from collections import deque
from typing import Callable, List, Optional, Tuple, Dict, Any

from syne_tune.config_space import (
    Domain,
    Integer,
    Float,
    FiniteRange,
    config_space_to_json_dict,
)
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import (
    SchedulerDecision,
    TrialSuggestion,
    TrialScheduler,
)
from syne_tune.config_space import cast_config_values, postprocess_config
from syne_tune.optimizer.schedulers.searchers.random_searcher import RandomSearcher
from syne_tune.util import dump_json_with_numpy

logger = logging.getLogger(__name__)


@dataclass
class PBTTrialState:
    """Internal PBT state tracked per-trial."""

    trial: Trial
    last_score: float = None
    last_checkpoint: int = None
    last_perturbation_time: int = 0
    stopped: bool = False


class PopulationBasedTraining(TrialScheduler):
    """
    Implements the Population Based Training (PBT) algorithm. This is an adapted
    version of the Ray Tune implementation:

    https://docs.ray.io/en/latest/tune/tutorials/tune-advanced-tutorial.html

    PBT was originally presented in the following paper:

        | Jaderberg et. al.
        | Population Based Training of Neural Networks
        | https://arxiv.org/abs/1711.09846

    Population based training (PBT) maintains a population of models spread across
    an asynchronous set of workers and dynamically adjust their hyperparameters
    during training. Every time a worker reaches a user-defined milestone, it
    returns the performance of the currently evaluated network. If the network is
    within the top percentile of the population, the worker resumes its training
    until the next milestone. If not, PBT selects a model from the top percentile
    uniformly at random. The worker now continues with the latest checkpoint of
    this new model but mutates the hyperparameters.

    The mutation happens as following. For each hyperparameter, we either resample
    its value uniformly at random, or otherwise increment (multiply by 1.2) or
    decrement (multiply by 0.8) the value (probability 0.5 each). For categorical
    hyperparameters, the value is always resampled uniformly.

    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of metric to optimize, key in results obtained via
       ``on_trial_result``.
    :param resource_attr: Name of resource attribute in results obtained
        via ``on_trial_result``, defaults to "time_total_s"
    :param max_t: max time units per trial. Trials will be stopped after
        ``max_t`` time units (determined by ``time_attr``) have passed.
        Defaults to 100
    :param custom_explore_fn: Custom exploration function. This
        function is invoked as ``f(config)`` instead of the built-in perturbations,
        and should return ``config`` updated as needed. If this is given,
        ``resample_probability`` is not used
    :param do_minimize: If True, we minimize the objective function specified by ``metric`` . Defaults to True.
    :param random_seed: Seed for initializing random number generators.
    :param population_size: Size of the population, defaults to 4
    :param perturbation_interval: Models will be considered for perturbation
        at this interval of ``resource_attr``. Note that perturbation incurs
        checkpoint overhead, so you shouldn't set this to be too frequent.
        Defaults to 60
    :param quantile_fraction: Parameters are transferred from the top
        ``quantile_fraction`` fraction of trials to the bottom
        ``quantile_fraction`` fraction. Needs to be between 0 and 0.5. Setting
        it to 0 essentially implies doing no exploitation at all.
        Defaults to 0.25
    :param resample_probability: The probability of resampling from the
        original distribution when applying :meth:`_explore`. If not
        resampled, the value will be perturbed by a factor of 1.2 or 0.8 if
        continuous, or changed to an adjacent value if discrete.
        Defaults to 0.25
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        resource_attr: str,
        max_t: int = 100,
        custom_explore_fn: Optional[Callable[[dict], dict]] = None,
        do_minimize: Optional[bool] = True,
        random_seed: int = None,
        population_size: int = 4,
        perturbation_interval: int = 60,
        quantile_fraction: float = 0.25,
        resample_probability: float = 0.25,
        searcher_kwargs: dict = None,
    ):
        super().__init__(random_seed=random_seed)

        # The current implementation only supports a random searcher
        self.metric = metric
        self.config_space = config_space

        if searcher_kwargs is None:
            self.searcher_kwargs = dict()
        else:
            self.searcher_kwargs = searcher_kwargs

        self.searcher = RandomSearcher(
            config_space=config_space,
            random_seed=self.random_seed,
            points_to_evaluate=self.searcher_kwargs.get("points_to_evaluate"),
        )
        self.resource_attr = resource_attr
        self.population_size = population_size
        self.perturbation_interval = perturbation_interval
        self.quantile_fraction = quantile_fraction
        self.resample_probability = resample_probability
        self.custom_explore_fn = custom_explore_fn
        self.max_t = max_t
        self.do_minimize = do_minimize
        self.metric_op = -1.0 if do_minimize else 1.0  # PBT assumes that we maximize
        self.trial_state = dict()
        self.next_perturbation_sync = self.perturbation_interval
        self.trial_decisions_stack = deque()
        self.num_checkpoints = 0
        self.num_perturbations = 0
        self.random_state = np.random.RandomState(self.random_seed)

    def on_trial_add(self, trial: Trial):
        self.trial_state[trial.trial_id] = PBTTrialState(trial=trial)

    def _get_trial_id_to_continue(self, trial: Trial) -> int:
        """Determine which trial to continue.

        Following the original PBT formulation if the trial is not in the top %n
        percent, we sample a trial uniformly at random from the upper quantile.

        :param trial: Trial at question right now
        :return: ID (int) of trial which should be continued inplace of ``trial``
        """
        lower_quantile, upper_quantile = self._quantiles()
        # If we are not in the upper quantile, we pause:
        trial_id = trial.trial_id
        if trial_id in lower_quantile:
            # sample random trial from upper quantile
            trial_id_to_clone = int(self.random_state.choice(upper_quantile))
            assert trial_id != trial_id_to_clone
            logger.debug(
                f"Trial {trial_id} is in lower quantile, replaced by {trial_id_to_clone}"
            )
            return trial_id_to_clone
        else:
            return trial_id

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        for name, value in [
            ("resource_attr", self.resource_attr),
            ("metric", self.metric),
        ]:
            if value not in result:
                err_msg = (
                    f"Cannot find {name} {value} in result {result}. Make sure "
                    "this attribute is reported by your training function"
                )
                raise RuntimeError(err_msg)

        trial_id = trial.trial_id
        cost = result[self.resource_attr]
        state = self.trial_state[trial_id]

        # Stop if we reached the maximum budget of this configuration
        if cost >= self.max_t:
            state.stopped = True
            return SchedulerDecision.STOP

        # Continue training if perturbation interval has not been reached yet.
        if cost - state.last_perturbation_time < self.perturbation_interval:
            return SchedulerDecision.CONTINUE

        self._save_trial_state(state, cost, result)

        state.last_perturbation_time = cost

        trial_id_to_continue = self._get_trial_id_to_continue(trial)

        if trial_id_to_continue == trial_id:
            # continue current trial
            return SchedulerDecision.CONTINUE
        else:
            # Note: At this point, the trial ``trial`` is marked as stopped, so it
            # cannot be proposed anymore in :meth:`_get_trial_id_to_continue` as
            # trial to continue from. This means we can just stop the trial, and
            # its checkpoint can be removed
            state.stopped = True
            # exploit step
            trial_to_clone = self.trial_state[trial_id_to_continue].trial

            # explore step
            config = self._explore(trial_to_clone.config)
            self.trial_decisions_stack.append((trial_id_to_continue, config))
            return SchedulerDecision.STOP

    def _save_trial_state(
        self, state: PBTTrialState, time: int, result: Dict[str, Any]
    ) -> float:
        """Saves necessary trial information when result is received.

        :param state: State object for trial.
        :param time: The current timestep (resource level) of the trial.
        :param result: The trial's result dictionary.
        :return: Current score
        """

        # This trial has reached its perturbation interval.
        # Record new state in the state object.
        score = self.metric_op * result[self.metric]
        state.last_score = score
        state.last_train_time = time
        state.last_result = result
        return score

    def _quantiles(self) -> Tuple[List[Trial], List[Trial]]:
        """Returns trials in the lower and upper ``quantile`` of the population.

        If there is not enough data to compute this, returns empty lists.

        :return ``(lower_quantile, upper_quantile)``
        """
        trials = []
        for trial, state in self.trial_state.items():
            if not state.stopped and state.last_score is not None:
                trials.append(trial)

        trials.sort(key=lambda t: self.trial_state[t].last_score)

        if len(trials) <= 1:
            return [], []
        else:
            num_trials_in_quantile = int(
                math.ceil(len(trials) * self.quantile_fraction)
            )
            if num_trials_in_quantile > len(trials) / 2:
                num_trials_in_quantile = int(math.floor(len(trials) / 2))
            return trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:]

    def suggest(self) -> Optional[TrialSuggestion]:
        if len(self.trial_decisions_stack) == 0:
            # If our stack is empty, we simply start a new random configuration.
            config = self.searcher.suggest()
            config = cast_config_values(config, self.config_space)
            return TrialSuggestion.start_suggestion(
                postprocess_config(config, self.config_space)
            )
        else:
            trial_id_to_continue, config = self.trial_decisions_stack.pop()
            config = cast_config_values(config=config, config_space=self.config_space)
            return TrialSuggestion.start_suggestion(
                config=config, checkpoint_trial_id=trial_id_to_continue
            )

    def _explore(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a config perturbed as specified.

        :param config: Original hyperparameter configuration from the cloned trial
        :return: Perturbed config
        """

        new_config = copy.deepcopy(config)

        self.num_perturbations += 1

        if self.custom_explore_fn:
            new_config = self.custom_explore_fn(new_config)
            assert (
                new_config is not None
            ), "Custom explore fn failed to return new config"
            return new_config

        for key, hp_range in self.config_space.items():
            if isinstance(hp_range, Domain):
                # For ``Categorical``, all values have the same distance from each
                # other, so we can always resample uniformly
                is_numerical = (
                    isinstance(hp_range, Float)
                    or isinstance(hp_range, Integer)
                    or isinstance(hp_range, FiniteRange)
                )
                if (
                    not is_numerical
                ) or self.random_state.rand() < self.resample_probability:
                    new_config[key] = hp_range.sample(
                        size=1, random_state=self.random_state
                    )
                else:
                    multiplier = 1.2 if self.random_state.rand() > 0.5 else 0.8
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

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata for the scheduler
        """
        metadata = super().metadata()
        config_space_json = dump_json_with_numpy(
            config_space_to_json_dict(self.config_space)
        )
        metadata["config_space"] = config_space_json
        metadata["metric_names"] = self.metric_names()
        metadata["metric_mode"] = self.metric_mode()
        return metadata

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return "min" if self.do_minimize else "max"
