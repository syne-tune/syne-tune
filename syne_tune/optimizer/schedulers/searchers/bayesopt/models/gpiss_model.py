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
from typing import Dict, List, Optional
import numpy as np
import logging

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer \
    import TransformerModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base \
    import BaseSurrogateModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext \
    import ExtendedConfiguration
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state \
    import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.gpiss_model \
    import GaussianProcessISSModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.issm \
    import prepare_data
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes \
    import SurrogateModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log \
    import DebugLogPrinter
from syne_tune.optimizer.schedulers.utils.simple_profiler \
    import SimpleProfiler
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import INTERNAL_METRIC_NAME

logger = logging.getLogger(__name__)


class GaussProcISSSurrogateModel(BaseSurrogateModel):
    def __init__(
            self, state: TuningJobState, active_metric: str,
            gpmodel: GaussianProcessISSModel,
            hp_ranges: HyperparameterRanges,
            means_observed_candidates: np.ndarray):
        """
        Gaussian Process Innovation State Space (GP-ISS) surrogate model, where
        model parameters are fit by marginal likelihood maximization.

        Pending evaluations in `state` are not taken into account here.
        Note that `state` contains extended configs (x, r), while the GP
        part of the GP-ISSM is over configs x (it models the function
        at r_max).

        :param state: TuningJobSubState
        :param active_metric: Name of the metric to optimize.
        :param gpmodel: GaussianProcessISSModel
        :param hp_ranges: HyperparameterRanges for predictions
        :param means_observed_candidates: Used for `current_best`
        """
        super().__init__(state, active_metric)
        self._gpmodel = gpmodel
        self._hp_ranges = hp_ranges
        self._means_observed_candidates = means_observed_candidates

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Input features `inputs` are w.r.t. configs x, not extended configs.
        Predictions are for f(x, r_max), at the maximum resource level.

        :param inputs: Input features
        :return: Predictive means, stddevs
        """
        predictions_list = []
        for post_mean, post_variance in self._gpmodel.predict(
                inputs):
            assert post_mean.shape[0] == inputs.shape[0], \
                (post_mean.shape, inputs.shape)
            assert post_variance.shape == (inputs.shape[0],), \
                (post_variance.shape, inputs.shape)
            post_std = np.sqrt(post_variance)
            predictions_list.append(
                {'mean': post_mean, 'std': post_std})
        return predictions_list

    def _hp_ranges_for_prediction(self) -> HyperparameterRanges:
        """
        For this model, predictions are done on normal configs, not on extended
        ones (as self.state.hp_ranges would do).
        """
        return self._hp_ranges

    def backward_gradient(
            self, input: np.ndarray,
            head_gradients: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        poster_states = self._gpmodel.states
        assert poster_states is not None, \
            "Cannot run backward_gradient without a posterior state"
        assert len(poster_states) == len(head_gradients), \
            "len(posterior_states) = {} != {} = len(head_gradients)".format(
                len(poster_states), len(head_gradients))
        return [
            poster_state.backward_gradient(input, head_gradient)
            for poster_state, head_gradient in zip(
                poster_states, head_gradients)]

    def predict_mean_current_candidates(self) -> List[np.ndarray]:
        return [self._means_observed_candidates.reshape((-1, 1))]


# TODO:
# DebugLogPrinter has parts which are specific to GaussProcSurrogateModel,
# these are not used here (does this go through?)
class GaussProcISSModelFactory(TransformerModelFactory):
    def __init__(
            self, gpmodel: GaussianProcessISSModel,
            active_metric: str = INTERNAL_METRIC_NAME,
            profiler: Optional[SimpleProfiler] = None,
            debug_log: Optional[DebugLogPrinter] = None):
        """
        Pending evaluations in `state` are not taken into account here.
        Note that `state` contains extended configs (x, r), while the GP
        part of the GP-ISSM is over configs x (it models the function
        at r_max).

        :param gpmodel: GaussianProcessISSModel
        :param active_metric: Name of the metric to optimize.

        """
        self._gpmodel = gpmodel
        self.active_metric = active_metric
        self._configspace_ext = None  # Set later
        self._debug_log = debug_log
        self._profiler = profiler

    def set_configspace_ext(self, configspace_ext: ExtendedConfiguration):
        """
        Delayed assignment of `configspace_ext`.

        :param configspace_ext: ExtendedConfiguration to deal with extended
            configs (x, r) in `state`
        """
        r_min, r_max = configspace_ext.resource_attr_range
        assert 0 < r_min < r_max, \
            f"r_min = {r_min}, r_max = {r_max}: Need 0 < r_min < r_max"
        self._configspace_ext = configspace_ext

    @property
    def debug_log(self) -> Optional[DebugLogPrinter]:
        return self._debug_log

    @property
    def profiler(self) -> Optional[SimpleProfiler]:
        return self._profiler

    def get_params(self):
        return self._gpmodel.get_params()

    def set_params(self, param_dict):
        self._gpmodel.set_params(param_dict)

    def model(self, state: TuningJobState, fit_params: bool) -> SurrogateModel:
        assert self._configspace_ext is not None, \
            "configspace_ext not assigned (use 'set_configspace_ext')"
        assert state.num_observed_cases(self.active_metric) > 0, \
            "Cannot compute posterior: state has no labeled datapoints " +\
            f"for metric {self.active_metric}"
        if self._debug_log is not None:
            self._debug_log.set_state(state)
        # Prepare data in format required by GP-ISSM
        data = prepare_data(state, self._configspace_ext, self.active_metric)

        if not fit_params:
            logger.info("Recomputing posterior state")
            self._gpmodel.recompute_states(data, profiler=self._profiler)
        else:
            logger.info(f"Fitting surrogate model for {self.active_metric}")
            self._gpmodel.fit(data, profiler=self._profiler)
        if self._debug_log is not None:
            self._debug_log.set_model_params(self.get_params())

        return GaussProcISSSurrogateModel(
            state=state,
            active_metric=self.active_metric,
            gpmodel=self._gpmodel,
            hp_ranges=self._configspace_ext.hp_ranges,
            means_observed_candidates=self._predict_mean_current_candidates(
                data))

    def _predict_mean_current_candidates(self, data: Dict) -> np.ndarray:
        """
        Returns the predictive mean (signal with key 'mean') at all current
        candidate configurations (both state.candidate_evaluations and
        state.pending_evaluations). Different to multi-task GP models, these
        means are over f(x, r_max) only.

        :return: List of predictive means
        """
        means, _ = self._gpmodel.predict(data['features'])[0]
        return means.reshape((-1, 1))
