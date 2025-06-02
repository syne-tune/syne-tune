import logging
from typing import Optional, Union, Dict, Any, List

import numpy as np
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import (
    TrialScheduler,
    SchedulerDecision,
    TrialSuggestion,
)
from syne_tune.optimizer.schedulers.searchers.last_value_multi_fidelity_searcher import (
    LastValueMultiFidelitySearcher,
)
from syne_tune.optimizer.schedulers.searchers.multi_fidelity_searcher import (
    IndependentMultiFidelitySearcher,
)
from syne_tune.optimizer.schedulers.multiobjective.multiobjective_priority import (
    MOPriority,
)
from syne_tune.util import dump_json_with_numpy
from syne_tune.config_space import (
    cast_config_values,
    config_space_to_json_dict,
    remove_constant_and_cast,
    postprocess_config,
)


logger = logging.getLogger(__name__)


class AsynchronousSuccessiveHalving(TrialScheduler):
    """
    Implements Asynchronous Successive Halving. This code is adapted from the RayTune implementation.

    References:

    Massively Parallel Hyperparameter Tuning
    L. Li and K. Jamieson and A. Rostamizadeh and K. Gonina and M. Hardt and B. Recht and A. Talwalkar
    arXiv:1810.05934 [cs.LG]

    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of metric to optimize, key in results obtained via
       ``on_trial_result``.
    :param do_minimize: If True, we minimize the objective function specified by ``metric`` . Defaults to True.
    :param searcher: Searcher object to sample configurations.
    :param time_attr: A training result attr to use for comparing time.
        Note that you can pass in something non-temporal such as
        ``training_iteration`` as a measure of progress, the only requirement
        is that the attribute should increase monotonically.
        Defaults to "training_iteration"
    :param max_t: max time units per trial. Trials will be stopped after
        ``max_t`` time units (determined by ``time_attr``) have passed.
        Defaults to 100
    :param grace_period: Only stop trials at least this old in time.
        The units are the same as the attribute named by ``time_attr``.
        Defaults to 1
    :param reduction_factor: Used to set halving rate and amount. This
        is simply a unit-less scalar. Defaults to 3
    :param brackets: Number of brackets. Each bracket has a different
        ``grace_period`` and number of rung levels. Defaults to 1
    :param random_seed: Seed for initializing random number generators.
    :param searcher_kwargs: Additional keyword arguments for the searcher.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        searcher: Optional[
            Union[str, IndependentMultiFidelitySearcher, LastValueMultiFidelitySearcher]
        ] = "random_search",
        time_attr: str = "training_iteration",
        max_t: int = 100,
        grace_period: int = 1,
        reduction_factor: float = 3,
        brackets: int = 1,
        random_seed: int = None,
        searcher_kwargs: dict = None,
    ):
        super().__init__(random_seed=random_seed)

        assert max_t > 0, "Max (time_attr) not valid!"
        assert max_t >= grace_period, "grace_period must be <= max_t!"
        assert grace_period > 0, "grace_period must be positive!"
        assert reduction_factor > 1, "reduction factor not valid!"
        assert brackets > 0, "brackets must be positive!"

        self.config_space = config_space
        self.do_minimize = do_minimize
        self.metric = metric
        if isinstance(searcher, str):
            if searcher_kwargs is None:
                searcher_kwargs = {}

            self.searcher = LastValueMultiFidelitySearcher(
                searcher=searcher,
                config_space=config_space,
                random_seed=random_seed,
                **searcher_kwargs,
            )
        else:
            self.searcher = searcher

        self.reduction_factor = reduction_factor
        self.max_t = max_t
        self.trial_info = {}  # Stores Trial -> Bracket

        # Tracks state for new trial add
        self.brackets = [
            Bracket(
                grace_period,
                max_t,
                reduction_factor,
                s,
            )
            for s in range(brackets)
        ]
        self.num_stopped = 0
        self.metric_multiplier = 1 if self.do_minimize else -1
        self.time_attr = time_attr

    def suggest(self) -> Optional[TrialSuggestion]:
        config = self.searcher.suggest()
        if config is not None:
            config = cast_config_values(config, self.config_space)
            config = TrialSuggestion.start_suggestion(
                postprocess_config(config, self.config_space)
            )
        return config

    def on_trial_add(self, trial: Trial):
        sizes = np.array([len(b.rungs) for b in self.brackets])
        probs = np.e ** (sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self.brackets), p=normalized)
        self.trial_info[trial.trial_id] = self.brackets[idx]

    def on_trial_error(self, trial: Trial):
        self.searcher.on_trial_error(trial.trial_id)
        logger.warning(f"trial_id {trial.trial_id}: Evaluation failed!")

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        config = remove_constant_and_cast(trial.config, self.config_space)
        metric = result[self.metric] * self.metric_multiplier
        self.searcher.on_trial_result(
            trial.trial_id, config, metric=metric, resource_level=result[self.time_attr]
        )
        self._check_metrics_are_present(result)
        if result[self.time_attr] >= self.max_t:
            action = SchedulerDecision.STOP
        else:
            bracket = self.trial_info[trial.trial_id]
            action = bracket.on_result(
                trial_id=trial.trial_id,
                cur_iter=result[self.time_attr],
                metrics={self.metric: metric},
            )
        if action == SchedulerDecision.STOP:
            self.num_stopped += 1
        return action

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):

        config = remove_constant_and_cast(trial.config, self.config_space)
        metric = result[self.metric] * self.metric_multiplier
        self.searcher.on_trial_result(
            trial.trial_id, config, metric=metric, resource_level=result[self.time_attr]
        )

        self._check_metrics_are_present(result)
        bracket = self.trial_info[trial.trial_id]
        bracket.on_result(
            trial_id=trial.trial_id,
            cur_iter=result[self.time_attr],
            metrics={self.metric: metric},
        )
        del self.trial_info[trial.trial_id]

    def on_trial_remove(self, trial: Trial):
        del self.trial_info[trial.trial_id]

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return "min" if self.do_minimize else "max"

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata for the scheduler
        """
        metadata = super().metadata()
        config_space_json = dump_json_with_numpy(
            config_space_to_json_dict(self.config_space)
        )
        metadata["config_space"] = config_space_json
        metadata["metric"] = self.metric
        metadata["metric_names"] = self.metric_names()
        metadata["metric_mode"] = self.metric_mode()
        return metadata

    def _check_metrics_are_present(self, result: Dict[str, Any]):
        for key in [self.metric, self.time_attr]:
            if key not in result:
                assert key in result, f"{key} not found in reported result {result}"


class Bracket:
    """Bookkeeping system to track recorded values.

    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.
    """

    def __init__(
        self,
        min_t: int,
        max_t: int,
        reduction_factor: float,
        s: int,
        priority: Optional[MOPriority] = None,
    ):
        self.rf = reduction_factor
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        self.rungs = [
            (min_t * self.rf ** (k + s), {}) for k in reversed(range(MAX_RUNGS))
        ]
        self.priority = priority

    def on_result(self, trial_id: int, cur_iter: int, metrics: Optional[dict]) -> str:
        action = SchedulerDecision.CONTINUE
        for milestone, recorded in self.rungs:
            if cur_iter < milestone or trial_id in recorded:
                continue
            else:
                if not recorded:
                    # if no result was previously recorded, we saw the first result and we continue
                    action = SchedulerDecision.CONTINUE
                else:
                    # get the list of metrics seen for the rung, compute rank and decide to continue
                    # if trial is in the top ones according to a rank induced by the ``reduction_factor``.
                    metric_recorded = np.array(
                        [list(x.values()) for x in recorded.values()]
                        + [list(metrics.values())]
                    )
                    if self.priority is not None:
                        priorities = self.priority(metric_recorded)
                    else:
                        # single objective case
                        priorities = metric_recorded.flatten()
                    ranks = np.searchsorted(sorted(priorities), priorities) / len(
                        priorities
                    )
                    new_priority_rank = ranks[-1]
                    if new_priority_rank > 1 / self.rf:
                        action = SchedulerDecision.STOP
                recorded[trial_id] = metrics
                break
        return action
