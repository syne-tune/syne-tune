from typing import List, Optional
import numpy as np
import logging

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    Predictor,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import ConfigurationFilter

logger = logging.getLogger(__name__)


class BasePredictor(Predictor):
    """
    Base class for (most)
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`
    implementations, provides common code.
    """

    def __init__(
        self,
        state: TuningJobState,
        active_metric: Optional[str] = None,
        filter_observed_data: Optional[ConfigurationFilter] = None,
    ):
        super().__init__(state, active_metric)
        self._current_best = None
        self._filter_observed_data = filter_observed_data

    @property
    def filter_observed_data(self) -> Optional[ConfigurationFilter]:
        return self._filter_observed_data

    def set_filter_observed_data(
        self, filter_observed_data: Optional[ConfigurationFilter]
    ):
        self._filter_observed_data = filter_observed_data
        self._current_best = None

    def predict_mean_current_candidates(self) -> List[np.ndarray]:
        """
        Returns the predictive mean (signal with key 'mean') at all current candidates
        in the state (observed, pending).

        If the hyperparameters of the surrogate model are being optimized (e.g.,
        by empirical Bayes), the returned list has length 1. If its
        hyperparameters are averaged over by MCMC, the returned list has one
        entry per MCMC sample.

        :return: List of predictive means
        """
        candidates, _ = self.state.observed_data_for_metric(self.active_metric)
        candidates += self.state.pending_configurations()
        candidates = self._current_best_filter_candidates(candidates)
        assert (
            len(candidates) > 0
        ), "Cannot predict means at current candidates with no candidates at all"
        inputs = self.hp_ranges_for_prediction().to_ndarray_matrix(candidates)
        all_means = []
        # Loop over MCMC samples (if any)
        for prediction in self.predict(inputs):
            means = prediction["mean"]
            if means.ndim == 1:  # In case of no fantasizing
                means = means.reshape((-1, 1))
            all_means.append(means)
        return all_means

    def current_best(self) -> List[np.ndarray]:
        if self._current_best is None:
            all_means = self.predict_mean_current_candidates()
            result = [np.min(means, axis=0) for means in all_means]
            self._current_best = result
        return self._current_best

    def _current_best_filter_candidates(self, candidates):
        """
        In some subclasses, 'current_best' is not computed over all (observed
        and pending) candidates: they need to implement this filter.
        """
        if self._filter_observed_data is None:
            return candidates  # Default: No filtering
        else:
            filtered_candidates = [
                config for config in candidates if self._filter_observed_data(config)
            ]
            return filtered_candidates
