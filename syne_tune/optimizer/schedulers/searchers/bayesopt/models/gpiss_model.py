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

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    TransformerModelFactory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base import (
    BaseSurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext import (
    ExtendedConfiguration,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.gpiss_model import (
    GaussianProcessLearningCurveModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.issm import (
    prepare_data,
    prepare_data_with_pending,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.posterior_state import (
    GaussProcAdditivePosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    SurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    FantasizedPendingEvaluation,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import ConfigurationFilter
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler

logger = logging.getLogger(__name__)


class GaussProcAdditiveSurrogateModel(BaseSurrogateModel):
    def __init__(
        self,
        state: TuningJobState,
        gpmodel: GaussianProcessLearningCurveModel,
        fantasy_samples: List[FantasizedPendingEvaluation],
        active_metric: str,
        filter_observed_data: Optional[ConfigurationFilter] = None,
        normalize_mean: float = 0.0,
        normalize_std: float = 1.0,
    ):
        """
        Gaussian Process additive surrogate model, where model parameters are
        fit by marginal likelihood maximization.

        Note: `predict_mean_current_candidates` calls `predict` for all
        observed and pending extended configs. This may not be exactly
        correct, because `predict` is not meant to be used for configs
        which have observations (it IS correct at r = r_max).

        `fantasy_samples` contains the sampled (normalized) target values for
        pending configs. Only `active_metric` target values are considered.
        The target values for a pending config are a flat vector.

        :param state: TuningJobSubState
        :param gpmodel: GaussianProcessLearningCurveModel
        :param fantasy_samples: See above
        :param active_metric: See parent class
        :param filter_observed_data: See parent class
        :param normalize_mean: Mean used to normalize targets
        :param normalize_std: Stddev used to normalize targets

        """
        super().__init__(state, active_metric, filter_observed_data)
        self._gpmodel = gpmodel
        self.mean = normalize_mean
        self.std = normalize_std
        self.fantasy_samples = fantasy_samples

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Input features `inputs` are w.r.t. extended configs (x, r).

        :param inputs: Input features
        :return: Predictive means, stddevs
        """
        predictions_list = []
        for post_mean, post_variance in self._gpmodel.predict(inputs):
            assert post_mean.shape[0] == inputs.shape[0], (
                post_mean.shape,
                inputs.shape,
            )
            assert post_variance.shape == (inputs.shape[0],), (
                post_variance.shape,
                inputs.shape,
            )
            # Undo normalization applied to targets
            mean_denorm = post_mean * self.std + self.mean
            std_denorm = np.sqrt(post_variance) * self.std
            predictions_list.append({"mean": mean_denorm, "std": std_denorm})
        return predictions_list

    def backward_gradient(
        self, input: np.ndarray, head_gradients: List[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        poster_states = self.posterior_states
        assert (
            poster_states is not None
        ), "Cannot run backward_gradient without a posterior state"
        assert len(poster_states) == len(
            head_gradients
        ), "len(posterior_states) = {} != {} = len(head_gradients)".format(
            len(poster_states), len(head_gradients)
        )
        return [
            poster_state.backward_gradient(input, head_gradient, self.mean, self.std)
            for poster_state, head_gradient in zip(poster_states, head_gradients)
        ]

    def does_mcmc(self):
        return False

    @property
    def posterior_states(self) -> Optional[List[GaussProcAdditivePosteriorState]]:
        return self._gpmodel.states


class GaussProcAdditiveModelFactory(TransformerModelFactory):
    def __init__(
        self,
        gpmodel: GaussianProcessLearningCurveModel,
        num_fantasy_samples: int,
        active_metric: str,
        config_space_ext: ExtendedConfiguration,
        normalize_targets: bool = False,
        profiler: Optional[SimpleProfiler] = None,
        debug_log: Optional[DebugLogPrinter] = None,
        filter_observed_data: Optional[ConfigurationFilter] = None,
    ):
        """
        If `num_fantasy_samples > 0`, we draw this many fantasy targets
        independently, while each sample is dependent over all pending
        evaluations. If `num_fantasy_samples == 0`, pending evaluations
        in `state` are ignored.

        :param gpmodel: GaussianProcessLearningCurveModel
        :param num_fantasy_samples: See above
        :param active_metric: Name of the metric to optimize.
        :param config_space_ext: ExtendedConfiguration
        :param normalize_targets: Normalize observed target values?
        :param debug_log: DebugLogPrinter (optional)
        :param filter_observed_data: Filter for observed data before
            computing incumbent

        """
        self._gpmodel = gpmodel
        self.active_metric = active_metric
        r_min, r_max = config_space_ext.resource_attr_range
        assert (
            0 < r_min < r_max
        ), f"r_min = {r_min}, r_max = {r_max}: Need 0 < r_min < r_max"
        assert (
            num_fantasy_samples >= 0
        ), f"num_fantasy_samples = {num_fantasy_samples}, must be non-negative int"
        self.num_fantasy_samples = num_fantasy_samples
        self._config_space_ext = config_space_ext
        self._debug_log = debug_log
        self._profiler = profiler
        self._filter_observed_data = filter_observed_data
        self.normalize_targets = normalize_targets

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
        assert state.num_observed_cases(self.active_metric) > 0, (
            "Cannot compute posterior: state has no labeled datapoints "
            + f"for metric {self.active_metric}"
        )
        if self._debug_log is not None:
            self._debug_log.set_state(state)
        do_fantasizing = state.pending_evaluations and self.num_fantasy_samples > 0

        # [1] Fit model and compute posterior state, ignoring pending evals
        data = prepare_data(
            state,
            self._config_space_ext,
            self.active_metric,
            normalize_targets=self.normalize_targets,
            do_fantasizing=False,
        )
        if fit_params:
            logger.info(f"Fitting surrogate model for {self.active_metric}")
            self._gpmodel.fit(data, profiler=self._profiler)
        elif not do_fantasizing:
            # Only if part below is skipped
            logger.info("Recomputing posterior state")
            self._gpmodel.recompute_states(data)
        if self._debug_log is not None:
            self._debug_log.set_model_params(self.get_params())
        if self.normalize_targets:
            extra_kwargs = {
                "normalize_mean": data["mean_targets"],
                "normalize_std": data["std_targets"],
            }
        else:
            extra_kwargs = dict()

        # [2] Fantasizing for pending evaluations (optional)
        if do_fantasizing:
            # Sample fantasy values for pending evaluations
            logger.info("Sampling fantasy target values for pending evaluations")
            state_with_fantasies = self._draw_fantasy_values(state)
            fantasy_samples = state_with_fantasies.pending_evaluations
            # Recompute posterior state with fantasy samples
            logger.info("Recomputing posterior state with fantasy targets")
            data = prepare_data(
                state=state_with_fantasies,
                config_space_ext=self._config_space_ext,
                active_metric=self.active_metric,
                normalize_targets=self.normalize_targets,
                do_fantasizing=True,
            )
            self._gpmodel.recompute_states(data)
        else:
            fantasy_samples = []

        return GaussProcAdditiveSurrogateModel(
            state=state,
            gpmodel=self._gpmodel,
            fantasy_samples=fantasy_samples,
            active_metric=self.active_metric,
            filter_observed_data=self._filter_observed_data,
            **extra_kwargs,
        )

    def model_for_fantasy_samples(
        self, state: TuningJobState, fantasy_samples: List[FantasizedPendingEvaluation]
    ) -> SurrogateModel:
        """
        Same as `model` with `fit_params=False`, but `fantasy_samples` are
        passed in, rather than sampled here.

        :param state: See `model`
        :param fantasy_samples: See above
        :return: See `model`

        """
        assert state.num_observed_cases(self.active_metric) > 0, (
            "Cannot compute posterior: state has no labeled datapoints "
            + f"for metric {self.active_metric}"
        )
        assert state.pending_evaluations and self.num_fantasy_samples > 0

        # Recompute posterior state with fantasy samples
        state_with_fantasies = TuningJobState(
            hp_ranges=state.hp_ranges,
            config_for_trial=state.config_for_trial,
            trials_evaluations=state.trials_evaluations,
            failed_trials=state.failed_trials,
            pending_evaluations=fantasy_samples,
        )
        # Recompute posterior state with fantasy samples
        data = prepare_data(
            state=state_with_fantasies,
            config_space_ext=self._config_space_ext,
            active_metric=self.active_metric,
            normalize_targets=self.normalize_targets,
            do_fantasizing=True,
        )
        self._gpmodel.recompute_states(data)
        if self.normalize_targets:
            extra_kwargs = {
                "normalize_mean": data["mean_targets"],
                "normalize_std": data["std_targets"],
            }
        else:
            extra_kwargs = dict()

        return GaussProcAdditiveSurrogateModel(
            state=state,
            gpmodel=self._gpmodel,
            fantasy_samples=fantasy_samples,
            active_metric=self.active_metric,
            filter_observed_data=self._filter_observed_data,
            **extra_kwargs,
        )

    def _draw_fantasy_values(self, state: TuningJobState) -> TuningJobState:
        """
        Note: Fantasized target values are not de-normalized, because they
        are used internally only (see `prepare_data` with
        `do_fantasizing=True`).

        :param state: State with pending evaluations without fantasies
        :return: Copy of `state`, where `pending_evaluations` contains
            fantasized target values

        """
        assert self.num_fantasy_samples > 0
        # Fantasies are drawn in sequential chunks, one trial with pending
        # evaluations at a time.
        data_nopending, data_pending = prepare_data_with_pending(
            state=state,
            config_space_ext=self._config_space_ext,
            active_metric=self.active_metric,
            normalize_targets=self.normalize_targets,
        )
        if not data_nopending["configs"]:
            # It can happen that all trials with observed data also have
            # pending evaluations. This is possible only at the very start,
            # as long as no trial has been stopped or paused.
            # In this case, we find the trial with the largest number of
            # observed targets and remove its pending evaluations, so
            # `data_nopending` gets one entry. It is not possible to compute
            # a posterior state without any data, so handling this case
            # correctly would be very tedious).
            assert data_pending[
                "configs"
            ], "State is empty, cannot do posterior inference:\n" + str(state)
            names = ("configs", "targets", "trial_ids")
            elem = {k: data_pending[k].pop(0) for k in names}
            for k, v in elem.items():
                data_nopending[k] = [v]
            k = "features"
            all_features = data_pending[k]
            data_nopending[k] = all_features[0].reshape((1, -1))
            data_pending[k] = all_features[1:, :]
            logger.info(
                "All trials currently have pending evaluations. In order to "
                "sample fantasy targets, I'll remove pending evaluations "
                f"from trial_id {elem['trial_ids']} (which has "
                f"{elem['targets'].size} observations)"
            )
        # Start with posterior state, conditioned on data from trials without
        # pending evaluations
        self._gpmodel.recompute_states(data_nopending)
        poster_state_nopending = self._gpmodel.states[0]
        # Loop over trials with pending evaluations: For each trial, we sample
        # fantasy targets given observed ones, then update `poster_state` by
        # conditioning on both. This ensures we obtain a joint sample (the
        # ordering of trials does not matter). For the application here, we
        # do not need the final `poster_state`.
        all_fantasy_targets = []
        for sample_it in range(self.num_fantasy_samples):
            fantasy_targets, _ = poster_state_nopending.sample_and_update_for_pending(
                data_pending,
                sample_all_nonobserved=False,
                random_state=self._gpmodel.random_state,
            )
            for pos, fantasies in enumerate(fantasy_targets):
                if sample_it == 0:
                    all_fantasy_targets.append([fantasies])
                else:
                    all_fantasy_targets[pos].append(fantasies)
        # Convert into `FantasizedPendingEvaluation`
        r_min = self._config_space_ext.resource_attr_range[0]
        pending_evaluations_with_fantasies = []
        for trial_id, targets, fantasies in zip(
            data_pending["trial_ids"], data_pending["targets"], all_fantasy_targets
        ):
            n_observed = targets.size
            n_pending = fantasies[0].size
            start = r_min + n_observed
            resources = list(range(start, start + n_pending))
            fantasy_matrix = np.hstack(v.reshape((-1, 1)) for v in fantasies)
            assert fantasy_matrix.shape == (n_pending, self.num_fantasy_samples)
            for resource, fantasy in zip(resources, fantasy_matrix):
                pending_evaluations_with_fantasies.append(
                    FantasizedPendingEvaluation(
                        trial_id=trial_id,
                        fantasies={self.active_metric: fantasy},
                        resource=resource,
                    )
                )
        # Return new state, with `pending_evaluations` replaced
        return TuningJobState(
            hp_ranges=state.hp_ranges,
            config_for_trial=state.config_for_trial,
            trials_evaluations=state.trials_evaluations,
            failed_trials=state.failed_trials,
            pending_evaluations=pending_evaluations_with_fantasies,
        )
