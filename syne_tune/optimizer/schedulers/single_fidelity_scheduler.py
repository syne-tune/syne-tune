from typing import Optional, Dict, Any, Union, List
import logging

from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import (
    cast_config_values,
    config_space_to_json_dict,
    remove_constant_and_cast,
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
    Scheduler class for both single- and multi-objective methods that optimize using a single fidelity only,
    e.g., the highest amount of resources.

    :param config_space: Configuration space for evaluation function
    :param metrics: Name of metric to optimize, key in results obtained via
        ``on_trial_result``.
    :param do_minimize: True if we minimize the objective function
    :param searcher: Searcher for ``get_config`` decisions. String values
        are passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory` along
        with ``search_options`` and extra information. Supported values:
        :const:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.SUPPORTED_SEARCHERS_FIFO`.
        Defaults to "random" (i.e., random search)
    :param random_seed: Seed for initializing random number generators.
    :param searcher_kwargs: Additional arguments for the searcher.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metrics: List[str],
        do_minimize: Optional[bool] = True,
        searcher: Optional[Union[str, BaseSearcher]] = "random_search",
        random_seed: int = None,
        searcher_kwargs: dict = None,
    ):
        super().__init__(random_seed=random_seed)

        self.metrics = metrics
        self.config_space = config_space
        self.do_minimize = do_minimize
        self.metric_multiplier = 1 if self.do_minimize else -1

        if isinstance(searcher, str):
            if searcher_kwargs is None:
                searcher_kwargs = {}

            self.searcher = searcher_factory(searcher, config_space, **searcher_kwargs)
        else:
            self.searcher = searcher

    def suggest(self) -> Optional[TrialSuggestion]:

        config = self.searcher.suggest()
        if config is not None:
            config = cast_config_values(config, self.config_space)
            config = TrialSuggestion.start_suggestion(
                postprocess_config(config, self.config_space)
            )
        return config

    def on_trial_error(self, trial: Trial):
        self.searcher.on_trial_error(trial.trial_id)
        logger.warning(f"trial_id {trial.trial_id}: Evaluation failed!")

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
        config = remove_constant_and_cast(trial.config, self.config_space)
        metric = [
            result[metric_name] * self.metric_multiplier for metric_name in self.metrics
        ]
        self.searcher.on_trial_result(trial.trial_id, config, metric)
        return SchedulerDecision.CONTINUE

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        """Notification for the completion of trial.

        Note that :meth:`on_trial_result` is called with the same result before.
        However, if the scheduler only uses one final report from each
        trial, it may ignore :meth:`on_trial_result` and just use ``result`` here.

        :param trial: Trial which is completing
        :param result: Result dictionary
        """
        config = remove_constant_and_cast(trial.config, self.config_space)
        metric = [
            result[metric_name] * self.metric_multiplier for metric_name in self.metrics
        ]
        self.searcher.on_trial_complete(trial.trial_id, config, metric)

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
        return self.metrics

    def metric_mode(self) -> str:
        return "min" if self.do_minimize else "max"
