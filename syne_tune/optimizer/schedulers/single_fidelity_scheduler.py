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
from typing import Optional, List, Dict, Any, Union
import logging

from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import (
    cast_config_values,
    config_space_to_json_dict,
    preprocess_config,
    postprocess_config,
)
from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher
from syne_tune.util import dump_json_with_numpy
from syne_tune.optimizer.scheduler import (
    TrialScheduler,
    TrialSuggestion,
    SchedulerDecision,
)
from syne_tune.optimizer.schedulers.searchers.searcher_factory import searcher_factory

logger = logging.getLogger(__name__)


class SingleFidelityScheduler(TrialScheduler):
    """
    Schedulers maintain and drive the logic of an experiment, making decisions
    which configs to evaluate in new trials, and which trials to stop early.

    Some schedulers support pausing and resuming trials. In this case, they
    also drive the decision when to restart a paused trial.

    :param config_space: Configuration space for evaluation function
    :type config_space: Dict[str, Any]
    :param searcher: Searcher for ``get_config`` decisions. String values
        are passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory` along
        with ``search_options`` and extra information. Supported values:
        :const:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.SUPPORTED_SEARCHERS_FIFO`.
        Defaults to "random" (i.e., random search)
    :type searcher: str or
        :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher`
    :param metric: Name of metric to optimize, key in results obtained via
        ``on_trial_result``. For multi-objective schedulers, this can also be a
        list
    :type metric: str or List[str]
    :param mode: "min" if ``metric`` is minimized, "max" if ``metric`` is
        maximized, defaults to "min". This can also be a list if ``metric`` is
        a list
    :type mode: str or List[str], optional
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list
        can be partially specified, or even be an empty dict. For each
        hyperparameter not specified, the default value is determined using
        a midpoint heuristic.
        If not given, this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
        Note: If ``searcher`` is of type :class:`BaseSearcher`,
        ``points_to_evaluate`` must be set there.
    :type points_to_evaluate: ``List[dict]``, optional
    :param random_seed: Master random seed. Generators used in the
        scheduler or searcher are seeded using :class:`RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    :param time_keeper: This will be used for timing here (see
        ``_elapsed_time``). The time keeper has to be started at the beginning
        of the experiment. If not given, we use a local time keeper here,
        which is started with the first call to :meth:`_suggest`. Can also be set
        after construction, with :meth:`set_time_keeper`.
        Note: If you use
        :class:`~syne_tune.backend.simulator_backend.SimulatorBackend`, you need
        to pass its ``time_keeper`` here.
    :type time_keeper: :class:`~syne_tune.backend.time_keeper.TimeKeeper`,
        optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        searcher: Optional[Union[str, BaseSearcher]] = "random_search",
        random_seed: int = None,
        searcher_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(random_seed=random_seed)

        self.metric = metric
        self.config_space = config_space
        self.do_minimize = do_minimize
        self.metric_op = 1 if self.do_minimize else -1

        if isinstance(searcher, str):
            if searcher_kwargs is None:
                searcher_kwargs = {}

            self.searcher = searcher_factory(searcher,
                                             config_space,
                                             **searcher_kwargs)
        else:
            self.searcher = searcher

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:

        trial_id = str(trial_id)
        config = self.searcher.get_config(trial_id=trial_id)
        if config is not None:
            config = cast_config_values(config, self.config_space)
            config = TrialSuggestion.start_suggestion(
                postprocess_config(config, self.config_space)
            )
        return config

    def on_trial_error(self, trial: Trial):
        trial_id = str(trial.trial_id)
        self.searcher.evaluation_failed(trial_id)
        logger.warning(f"trial_id {trial_id}: Evaluation failed!")

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        """Called on each intermediate result reported by a trial.

        At this point, the trial scheduler can make a decision by returning
        one of :const:`SchedulerDecision.CONTINUE`,
        :const:`SchedulerDecision.PAUSE`, or :const:`SchedulerDecision.STOP`.
        This will only be called when the trial is currently running.

        :param trial: Trial for which results are reported
        :param result: Result dictionary
        :return: Decision what to do with the trial
        """
        config = preprocess_config(trial.config, self.config_space)
        observation = result[self.metric] * self.metric_op
        self.searcher.on_trial_result(
            str(trial.trial_id), config, observation=observation, update=False
        )
        return SchedulerDecision.CONTINUE

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        """Notification for the completion of trial.

        Note that :meth:`on_trial_result` is called with the same result before.
        However, if the scheduler only uses one final report from each
        trial, it may ignore :meth:`on_trial_result` and just use ``result`` here.

        :param trial: Trial which is completing
        :param result: Result dictionary
        """
        config = preprocess_config(trial.config, self.config_space)
        observation = result[self.metric] * self.metric_op
        self.searcher.on_trial_result(
            str(trial.trial_id), config, observation=observation, update=True
        )

    def metadata(self) -> Dict[str, Any]:
        """
        :return: Metadata for the scheduler
        """
        metadata = super().metadata()
        config_space_json = dump_json_with_numpy(
            config_space_to_json_dict(self.config_space)
        )
        metadata["config_space"] = config_space_json
        return metadata
