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
from typing import Optional, Union, List, Dict, Any

import numpy as np
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import (
    TrialScheduler,
    SchedulerDecision,
    TrialSuggestion,
)
from syne_tune.util import dump_json_with_numpy
from syne_tune.optimizer.schedulers.searchers import BaseSearcher
from syne_tune.config_space import (
    cast_config_values,
    config_space_to_json_dict,
    preprocess_config,
    postprocess_config,
)
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_factory


logger = logging.getLogger(__name__)


class AsynchronousSuccessiveHalving(TrialScheduler):
    """
    Implements Asynchronous Successive Halving
    References:

    :param config_space: Configuration space
    :param metric:
    :param mode: One of :code:`{"min", "max"}` or a list of these values (same
        size as ``metrics``). Determines whether objectives are minimized or
        maximized. Defaults to "min"
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
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        mode: Optional[str] = "min",
        searcher: Optional[Union[str, BaseSearcher]] = "random_search",
        time_attr: str = "training_iteration",
        max_t: int = 100,
        grace_period: int = 1,
        reduction_factor: float = 3,
        brackets: int = 1,
        random_seed: int = None,
        searcher_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(random_seed=random_seed)

        assert max_t > 0, "Max (time_attr) not valid!"
        assert max_t >= grace_period, "grace_period must be <= max_t!"
        assert grace_period > 0, "grace_period must be positive!"
        assert reduction_factor > 1, "reduction factor not valid!"
        assert brackets > 0, "brackets must be positive!"

        self.config_space = config_space
        self.mode = mode
        self.metric = metric
        if isinstance(searcher, str):
            if searcher_kwargs is None:
                searcher_kwargs = {
                    "mode": self.mode,
                    "metric": self.metric,
                    "config_space": self.config_space,
                }

            assert (
                "metric" in searcher_kwargs
            ), "Key 'metric' needs to be in searcher_kwargs"
            assert (
                "config_space" in searcher_kwargs
            ), "Key 'config_space' needs to be in searcher_kwargs"

            self.searcher = searcher_factory(searcher, **searcher_kwargs)
        else:
            self.searcher = searcher

        self._reduction_factor = reduction_factor
        self._max_t = max_t
        self._trial_info = {}  # Stores Trial -> Bracket

        # Tracks state for new trial add
        self._brackets = [
            _Bracket(
                grace_period,
                max_t,
                reduction_factor,
                s,
            )
            for s in range(brackets)
        ]
        self._num_stopped = 0
        self._metric_op = 1 if self.mode == "min" else -1
        self._time_attr = time_attr

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:

        trial_id = str(trial_id)
        config = self.searcher.get_config(trial_id=trial_id)
        if config is not None:
            config = cast_config_values(config, self.config_space)
            config = TrialSuggestion.start_suggestion(
                postprocess_config(config, self.config_space)
            )
        return config

    def on_trial_add(self, trial: Trial):
        sizes = np.array([len(b._rungs) for b in self._brackets])
        probs = np.e ** (sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self._brackets), p=normalized)
        self._trial_info[trial.trial_id] = self._brackets[idx]

    def on_trial_error(self, trial: Trial):
        trial_id = str(trial.trial_id)
        self.searcher.evaluation_failed(trial_id)
        logger.warning(f"trial_id {trial_id}: Evaluation failed!")

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        config = preprocess_config(trial.config, self.config_space)
        self.searcher.on_trial_result(
            str(trial.trial_id), config, result=result, update=False
        )
        self._check_metrics_are_present(result)
        if result[self._time_attr] >= self._max_t:
            action = SchedulerDecision.STOP
        else:
            bracket = self._trial_info[trial.trial_id]
            action = bracket.on_result(
                trial_id=trial.trial_id,
                cur_iter=result[self._time_attr],
                metric=result[self.metric] * self._metric_op,
            )
        if action == SchedulerDecision.STOP:
            self._num_stopped += 1
        return action

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):

        config = preprocess_config(trial.config, self.config_space)
        self.searcher.on_trial_result(
            str(trial.trial_id), config, result=result, update=True
        )

        self._check_metrics_are_present(result)
        bracket = self._trial_info[trial.trial_id]
        bracket.on_result(
            trial_id=trial.trial_id,
            cur_iter=result[self._time_attr],
            metric=result[self.metric] * self._metric_op,
        )
        del self._trial_info[trial.trial_id]

    def on_trial_remove(self, trial: Trial):
        del self._trial_info[trial.trial_id]

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata for the scheduler
        """
        metadata = super().metadata()
        config_space_json = dump_json_with_numpy(
            config_space_to_json_dict(self.config_space)
        )
        metadata["config_space"] = config_space_json
        metadata["mode"] = self.mode
        metadata["metric"] = self.metric

        return metadata

    def _check_metrics_are_present(self, result: Dict[str, Any]):
        for key in [self.metric, self._time_attr]:
            if key not in result:
                assert key in result, f"{key} not found in reported result {result}"


class _Bracket:
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
    ):
        self.rf = reduction_factor
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        self._rungs = [
            (min_t * self.rf ** (k + s), {}) for k in reversed(range(MAX_RUNGS))
        ]

    def priority(self, metric_recorded: np.array):
        return np.argsort(metric_recorded)

    def on_result(self, trial_id: int, cur_iter: int, metric: Optional[float]) -> str:
        action = SchedulerDecision.CONTINUE
        for milestone, recorded in self._rungs:
            if cur_iter < milestone or trial_id in recorded:
                continue
            else:
                if not recorded:
                    # if no result was previously recorded, we saw the first result and we continue
                    action = SchedulerDecision.CONTINUE
                else:
                    # get the list of metrics seen for the rung, compute rank and decide to continue
                    # if trial is in the top ones according to a rank induced by the ``reduction_factor``.
                    metric_recorded = np.array(list(recorded.values()) + [metric])
                    ranks = np.argsort(metric_recorded)
                    new_priority_rank = ranks[-1]
                    if new_priority_rank > 1 / self.rf:
                        action = SchedulerDecision.STOP
                recorded[trial_id] = metric
                break
        return action


from datetime import datetime
from syne_tune.config_space import randint


max_steps = 10

config_space = {
    "steps": max_steps,
    "width": randint(0, 20),
}
time_attr = "step"
metric = "mean_loss"


def make_trial(trial_id: int):
    return Trial(
        trial_id=trial_id,
        config={"steps": 0, "width": trial_id},
        creation_time=datetime.now(),
    )


scheduler = AsynchronousSuccessiveHalving(
    max_t=max_steps,
    brackets=1,
    reduction_factor=2.0,
    time_attr="step",
    metric=metric,
    config_space=config_space,
)

trial1 = make_trial(trial_id=0)
trial2 = make_trial(trial_id=1)
trial3 = make_trial(trial_id=2)

scheduler.on_trial_add(trial=trial1)
scheduler.on_trial_add(trial=trial2)
scheduler.on_trial_add(trial=trial3)

make_metric = lambda x: {time_attr: 1, metric: x}
decision1 = scheduler.on_trial_result(trial1, make_metric(2.0))
decision2 = scheduler.on_trial_result(trial2, make_metric(1.0))
decision3 = scheduler.on_trial_result(trial3, make_metric(3.0))
assert decision1 == SchedulerDecision.CONTINUE
assert decision2 == SchedulerDecision.CONTINUE
assert decision3 == SchedulerDecision.STOP
