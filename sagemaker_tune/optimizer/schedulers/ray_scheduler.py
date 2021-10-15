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
from typing import Dict, Optional, Union, List
import logging

from sagemaker_tune.optimizer.scheduler import TrialScheduler, \
    TrialSuggestion
from sagemaker_tune.backend.trial_status import Trial
from sagemaker_tune.search_space import Domain, value_type_and_transform

__all__ = ['RayTuneScheduler']

logger = logging.getLogger(__name__)


class RayTuneScheduler(TrialScheduler):
    from ray.tune.schedulers import FIFOScheduler as RT_FIFOScheduler
    from ray.tune.suggest import Searcher as RT_Searcher

    class RandomSearch(RT_Searcher):
        def __init__(self, config_space: Dict, points_to_evaluate: List[Dict], mode: str):
            super().__init__(mode=mode)
            self.config_space = config_space
            self._points_to_evaluate = points_to_evaluate

        def _next_initial_config(self) -> Optional[Dict]:
            if self._points_to_evaluate:
                return self._points_to_evaluate.pop(0)
            else:
                return None  # No more initial configs

        def suggest(self, trial_id: str) -> Optional[Dict]:
            config = self._next_initial_config()
            if config is None:
                config = {
                    k: v.sample() if hasattr(v, "sample") else v
                    for k, v in self.config_space.items()}
            return config

        def on_trial_complete(
                self, trial_id: str, result: Optional[Dict] = None,
                error: bool = False):
            pass

    def __init__(
            self,
            config_space: Dict,
            ray_scheduler=None,
            ray_searcher: Optional[RT_Searcher] = None,
            points_to_evaluate: Optional[List[Dict]] = None):
        """
        Allow to use Ray scheduler and searcher. Any searcher/scheduler should
        work, except such which need access to TrialRunner (e.g., PBT), this
        feature is not implemented yet.

        If `ray_searcher` is not given (defaults to random searcher), initial
        configurations to evaluate can be passed in `points_to_evaluate`. If
        `ray_searcher` is given, this argument is ignored (needs to be passed
        to `ray_searcher` at construction). Note: Use

        sagemaker_tune.optimizer.schedulers.searchers.impute_points_to_evaluate

        in order to preprocess `points_to_evaluate` specified by the user or
        the benchmark.

        :param config_space: configuration of the sampled space, for instance
        ```python
        hyperparameters = {
            "steps": max_steps,
            "width": uniform(0, 20),
            "height": uniform(-100, 100),
            "activation": choice(["relu", "tanh"])
        }
        ```
        :param ray_scheduler: Ray scheduler, defaults to FIFO scheduler
        :param ray_searcher: Ray searcher, defaults to random search
        :param points_to_evaluate: See above
        """
        super().__init__(config_space)
        if ray_scheduler is None:
            ray_scheduler = self.RT_FIFOScheduler()
        self.scheduler = ray_scheduler

        if ray_searcher is not None:
            self.mode = ray_searcher.mode
        else:
            if hasattr(ray_scheduler, "_mode"):
                self.mode = ray_scheduler._mode
            else:
                self.mode = 'min'

        if ray_searcher is None:
            ray_searcher = self.RandomSearch(
                config_space=self.convert_config_space(config_space),
                points_to_evaluate=points_to_evaluate,
                mode=self.mode,
            )

        elif points_to_evaluate is not None:
            logger.warning(
                "points_to_evaluate specified here will not be used. Pass this"
                " argument when creating ray_searcher")
        self.searcher = ray_searcher
        # todo this one is not implemented yet, PBT would require it
        self.trial_runner_wrapper = None

        if self.searcher.metric is not None and self.scheduler.metric is not None:
            assert self.scheduler.metric == self.searcher.metric, "searcher and scheduler must have the same metric."

    def on_trial_add(self, trial: Trial):
        self.scheduler.on_trial_add(
            trial_runner=self.trial_runner_wrapper,
            trial=trial,
        )

    def on_trial_error(self, trial: Trial):
        self.scheduler.on_trial_error(
            trial_runner=self.trial_runner_wrapper,
            trial=trial,
        )

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
        self._check_valid_result(result=result)
        self.searcher.on_trial_result(
            trial_id=str(trial.trial_id), result=result)
        return self.scheduler.on_trial_result(
            trial_runner=self.trial_runner_wrapper,
            trial=trial,
            result=result)

    def on_trial_complete(self, trial: Trial, result: Dict):
        self._check_valid_result(result=result)
        self.searcher.on_trial_complete(
            trial_id=str(trial.trial_id), result=result)
        self.scheduler.on_trial_complete(
            trial_runner=self.trial_runner_wrapper,
            trial=trial,
            result=result)

    def _check_valid_result(self, result: Dict):
        for m in self.metric_names():
            assert m in result, f"metric {m} is not present in reported results {result}," \
                                f" the metrics present when calling `report(...)` in your training functions should" \
                                f" be identical to the ones passed as metrics/time_attr to the scheduler and searcher"

    def on_trial_remove(self, trial: Trial):
        return self.scheduler.on_trial_remove(
            trial_runner=self.trial_runner_wrapper,
            trial=trial)

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        config = self.searcher.suggest(trial_id=str(trial_id))
        return TrialSuggestion.start_suggestion(config)

    def metric_names(self) -> List[str]:
        return [self.scheduler.metric]

    def metric_mode(self) -> str:
        return self.mode

    @staticmethod
    def convert_config_space(config_space):
        """
        Converts config_space from our type to the one of Ray Tune.

        Note: `randint(lower, upper)` in Ray Tune has exclusive `upper`, while
        this is inclusive for us. On the other hand, `lograndint(lower, upper)`
        has inclusive `upper` in Ray Tune as well.

        :param config_space:
        :return:
        """
        from ray import tune

        ray_config_space = dict()
        for name, hp_range in config_space.items():
            ray_val = hp_range
            if isinstance(hp_range, Domain):
                tp, is_log = value_type_and_transform(hp_range, name=name)
                if tp == float:
                    if is_log:
                        ray_val = tune.loguniform(
                            hp_range.lower, hp_range.upper)
                    else:
                        ray_val = tune.uniform(hp_range.lower, hp_range.upper)
                elif tp == int:
                    if is_log:
                        ray_val = tune.lograndint(
                            hp_range.lower, hp_range.upper)
                    else:
                        # Note: `tune.randint` has exclusive upper!
                        ray_val = tune.randint(
                            hp_range.lower, hp_range.upper + 1)
                else:
                    ray_val = tune.choice(hp_range.categories)
            ray_config_space[name] = ray_val
        return ray_config_space
