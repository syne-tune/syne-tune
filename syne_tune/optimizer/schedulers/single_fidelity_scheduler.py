from typing import Optional, Dict, Any, Union
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
    #TODO: Update docstring

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
        ``on_trial_result``.
    :type metric: str
    :param random_seed: Master random seed. Generators used in the
        scheduler or searcher are seeded using :class:`RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        searcher: Optional[Union[str, BaseSearcher]] = "random_search",
        random_seed: int = None,
        searcher_kwargs: dict = None,
    ):
        super().__init__(random_seed=random_seed)

        self.metric = metric
        self.config_space = config_space
        self.do_minimize = do_minimize
        self.metric_multiplier = 1 if self.do_minimize else -1

        if isinstance(searcher, str):
            if searcher_kwargs is None:
                searcher_kwargs = {}

            self.searcher = searcher_factory(searcher, config_space, **searcher_kwargs)
        else:
            self.searcher = searcher

    def suggest(self, trial_id: int) -> Optional[TrialSuggestion]:

        config = self.searcher.suggest(trial_id=trial_id)
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
        metric = result[self.metric] * self.metric_multiplier
        self.searcher.on_trial_result(
            trial.trial_id, config, metric=metric, update=False
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
        config = remove_constant_and_cast(trial.config, self.config_space)
        metric = result[self.metric] * self.metric_multiplier
        self.searcher.on_trial_result(
            trial.trial_id, config, metric=metric, update=True
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
