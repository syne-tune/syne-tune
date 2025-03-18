from typing import Dict, Any
import logging

from syne_tune.backend.trial_status import Trial
from syne_tune.results_callback import StoreResultsCallback
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.legacy_fifo import LegacyFIFOScheduler
from syne_tune.optimizer.schedulers.searchers import ModelBasedSearcher
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    ModelStateTransformer,
)

logger = logging.getLogger(__name__)


def _get_model_based_searcher(tuner):
    searcher = None
    scheduler = tuner.scheduler
    if isinstance(scheduler, LegacyFIFOScheduler):
        if isinstance(scheduler.searcher, ModelBasedSearcher):
            searcher = scheduler.searcher
            state_transformer = searcher.state_transformer
            if state_transformer is not None:
                if not isinstance(state_transformer, ModelStateTransformer):
                    searcher = None
                elif not state_transformer.use_single_model:
                    logger.warning(
                        "StoreResultsAndModelParamsCallback does not currently "
                        "support multi-model setups. Model parameters sre "
                        "not logged."
                    )
                    searcher = None
            else:
                searcher = None
    return searcher


def _extended_result(searcher, result):
    if searcher is not None:
        kwargs = dict()
        # Append surrogate model parameters to ``result``
        params = searcher.state_transformer.get_params()
        if params:
            prefix = "model_"
            kwargs = {prefix + k: v for k, v in params.items()}
        kwargs["cumulative_get_config_time"] = searcher.cumulative_get_config_time
        result = dict(result, **kwargs)
    return result


class StoreResultsAndModelParamsCallback(StoreResultsCallback):
    """
    Extends :class:`~syne_tune.tuner_callback.StoreResultsCallback` by also
    storing the current surrogate model parameters in :meth:`on_trial_result`.
    This works for schedulers with model-based searchers. For other schedulers,
    this callback behaves the same as the superclass.
    """

    def __init__(
        self,
        add_wallclock_time: bool = True,
    ):
        super().__init__(add_wallclock_time)
        self._searcher = None

    def on_tuning_start(self, tuner):
        super().on_tuning_start(tuner)
        self._searcher = _get_model_based_searcher(tuner)

    def on_trial_result(
        self, trial: Trial, status: str, result: Dict[str, Any], decision: str
    ):
        result = _extended_result(self._searcher, result)
        super().on_trial_result(trial, status, result, decision)


class SimulatorAndModelParamsCallback(SimulatorCallback):
    """
    Extends
    :class:`~syne_tune.backend.simulator_backend.simulator_callback.SimulatorCallback`
    by also storing the current surrogate model parameters in
    :meth:`on_trial_result`. This works for schedulers with model-based searchers.
    For other schedulers, this callback behaves the same as the superclass.
    """

    def __init__(self):
        super().__init__()
        self._searcher = None

    def on_tuning_start(self, tuner):
        super().on_tuning_start(tuner)
        self._searcher = _get_model_based_searcher(tuner)

    def on_trial_result(
        self, trial: Trial, status: str, result: Dict[str, Any], decision: str
    ):
        result = _extended_result(self._searcher, result)
        super().on_trial_result(trial, status, result, decision)
