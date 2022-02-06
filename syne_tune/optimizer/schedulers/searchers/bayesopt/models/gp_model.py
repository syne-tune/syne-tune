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
from typing import Dict, List, Optional, Union
import numpy as np
import logging
from dataclasses import dataclass

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    TransformerModelFactory,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base import (
    BaseSurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    FantasizedPendingEvaluation,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import ConfigurationFilter
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_regression import (
    GaussianProcessRegression,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gpr_mcmc import (
    GPRegressionMCMC,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.gpind_model import (
    IndependentGPPerResourceModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.hypertune.gp_model import (
    HyperTuneIndependentGPModel,
    HyperTuneJointGPModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    PosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    SurrogateModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler

logger = logging.getLogger(__name__)


GPModel = Union[
    GaussianProcessRegression,
    GPRegressionMCMC,
    IndependentGPPerResourceModel,
    HyperTuneIndependentGPModel,
    HyperTuneJointGPModel,
]


class GaussProcSurrogateModel(BaseSurrogateModel):
    """
    Gaussian process surrogate model, where model parameters are either fit by
    marginal likelihood maximization (`GaussianProcessRegression`), or
    integrated out by MCMC sampling (`GPRegressionMCMC`).
    """

    def __init__(
        self,
        state: TuningJobState,
        gpmodel: GPModel,
        fantasy_samples: List[FantasizedPendingEvaluation],
        active_metric: str = None,
        normalize_mean: float = 0.0,
        normalize_std: float = 1.0,
        filter_observed_data: Optional[ConfigurationFilter] = None,
        hp_ranges_for_prediction: Optional[HyperparameterRanges] = None,
    ):
        """
        Both `state` and `gpmodel` are immutable. If parameters of the latter
        are to be fit, this has to be done before.

        `fantasy_samples` contains the sampled (normalized) target values for
        pending configs. Only `active_metric` target values are considered.
        The target values for a pending config are a flat vector. If MCMC is
        used, its length is a multiple of the number of MCMC samples,
        containing the fantasy values for MCMC sample 0, sample 1, ...

        :param state: TuningJobSubState
        :param gpmodel: GPModel. Model parameters must have been fit and/or
            posterior states been computed
        :param fantasy_samples: See above
        :param active_metric: Name of the metric to optimize.
        :param normalize_mean: Mean used to normalize targets
        :param normalize_std: Stddev used to normalize targets
        """
        super().__init__(state, active_metric, filter_observed_data)
        self._gpmodel = gpmodel
        self.mean = normalize_mean
        self.std = normalize_std
        self.fantasy_samples = fantasy_samples
        self._hp_ranges_for_prediction = hp_ranges_for_prediction

    def hp_ranges_for_prediction(self) -> HyperparameterRanges:
        if self._hp_ranges_for_prediction is not None:
            return self._hp_ranges_for_prediction
        else:
            return super().hp_ranges_for_prediction()

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
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
        return isinstance(self._gpmodel, GPRegressionMCMC)

    @property
    def posterior_states(self) -> Optional[List[PosteriorState]]:
        return self._gpmodel.states

    def _current_best_filter_candidates(self, candidates):
        candidates = super()._current_best_filter_candidates(candidates)
        hp_ranges = self.state.hp_ranges
        candidates = hp_ranges.filter_for_last_pos_value(candidates)
        assert candidates, (
            "state.hp_ranges does not contain any candidates "
            + "(labeled or pending) with resource attribute "
            + "'{}' = {}".format(hp_ranges.name_last_pos, hp_ranges.value_for_last_pos)
        )
        return candidates


@dataclass
class InternalCandidateEvaluations:
    features: np.ndarray
    targets: np.ndarray
    mean: float
    std: float


# Note: If state.pending_evaluations is not empty, it must contain entries
# of type FantasizedPendingEvaluation, which contain the fantasy samples. This
# is the case only for internal states.
def get_internal_candidate_evaluations(
    state: TuningJobState,
    active_metric: str,
    normalize_targets: bool,
    num_fantasy_samples: int,
) -> InternalCandidateEvaluations:
    candidates, evaluation_values = state.observed_data_for_metric(
        metric_name=active_metric
    )
    hp_ranges = state.hp_ranges
    features = hp_ranges.to_ndarray_matrix(candidates)
    # Normalize
    # Note: The fantasy values in state.pending_evaluations are sampled
    # from the model fit to normalized targets, so they are already
    # normalized
    targets = np.vstack(evaluation_values).reshape((-1, 1))
    mean = 0.0
    std = 1.0
    if normalize_targets:
        std = max(np.std(targets).item(), 1e-9)
        mean = np.mean(targets).item()
        targets = (targets - mean) / std
    if state.pending_evaluations:
        # In this case, y becomes a matrix, where the observed values are
        # broadcast
        cand_lst = [
            hp_ranges.to_ndarray(config) for config in state.pending_configurations()
        ]
        fanta_lst = []
        for pending_eval in state.pending_evaluations:
            assert isinstance(
                pending_eval, FantasizedPendingEvaluation
            ), "state.pending_evaluations has to contain FantasizedPendingEvaluation"
            fantasies = pending_eval.fantasies[active_metric]
            assert (
                fantasies.size == num_fantasy_samples
            ), "All state.pending_evaluations entries must have length {}".format(
                num_fantasy_samples
            )
            fanta_lst.append(fantasies.reshape((1, -1)))
        targets = np.vstack([targets * np.ones((1, num_fantasy_samples))] + fanta_lst)
        features = np.vstack([features] + cand_lst)
    return InternalCandidateEvaluations(features, targets, mean, std)


class GaussProcModelFactory(TransformerModelFactory):
    def __init__(
        self,
        gpmodel: GPModel,
        active_metric: str,
        normalize_targets: bool = True,
        profiler: Optional[SimpleProfiler] = None,
        debug_log: Optional[DebugLogPrinter] = None,
        filter_observed_data: Optional[ConfigurationFilter] = None,
        no_fantasizing: bool = False,
        hp_ranges_for_prediction: Optional[HyperparameterRanges] = None,
    ):
        """
        We support pending evaluations via fantasizing. Note that state does
        not contain the fantasy values, but just the pending configs. Fantasy
        values are sampled here.

        :param gpmodel: GPModel model
        :param active_metric: Name of the metric to optimize.
        :param normalize_targets: Normalize observed target values?
        :param debug_log: DebugLogPrinter (optional)
        :param filter_observed_data: Filter for observed data before
            computing incumbent
        :param no_fantasizing: If True, pending evaluations in the state are
            simply ignored, fantasizing is not done (not recommended)
        :param hp_ranges_for_prediction: If given, `GaussProcSurrogateModel`
            should use this instead of `state.hp_ranges`

        """
        self._gpmodel = gpmodel
        self.active_metric = active_metric
        self.normalize_targets = normalize_targets
        self._debug_log = debug_log
        self._profiler = profiler
        self._filter_observed_data = filter_observed_data
        self._no_fantasizing = no_fantasizing
        self._hp_ranges_for_prediction = hp_ranges_for_prediction
        self._mean = None
        self._std = None

    @property
    def debug_log(self) -> Optional[DebugLogPrinter]:
        return self._debug_log

    @property
    def profiler(self) -> Optional[SimpleProfiler]:
        return self._profiler

    @property
    def gpmodel(self) -> GPModel:
        return self._gpmodel

    def model(self, state: TuningJobState, fit_params: bool) -> SurrogateModel:
        """
        Parameters of `self._gpmodel` are optimized iff `fit_params`. This
        requires `state` to contain labeled examples.

        If self.state.pending_evaluations is not empty, we proceed as follows:
        - Compute posterior for state without pending evals
        - Draw fantasy values for pending evals
        - Recompute posterior (without fitting)

        """
        if self._debug_log is not None:
            self._debug_log.set_state(state)
        # Compute posterior for state without pending evals
        no_pending_state = state
        if state.pending_evaluations:
            no_pending_state = TuningJobState(
                hp_ranges=state.hp_ranges,
                config_for_trial=state.config_for_trial,
                trials_evaluations=state.trials_evaluations,
                failed_trials=state.failed_trials,
            )
        self._posterior_for_state(
            no_pending_state, fit_params=fit_params, profiler=self._profiler
        )
        if state.pending_evaluations and not self._no_fantasizing:
            # Sample fantasy values for pending evaluations
            state_with_fantasies = self._draw_fantasy_values(state)
            # Compute posterior for state with pending evals
            # Note: profiler is not passed here, this would overwrite the
            # results from the first call
            self._posterior_for_state(
                state_with_fantasies, fit_params=False, profiler=None
            )
            fantasy_samples = state_with_fantasies.pending_evaluations
        else:
            fantasy_samples = []
        return GaussProcSurrogateModel(
            state=state,
            active_metric=self.active_metric,
            gpmodel=self._gpmodel,
            fantasy_samples=fantasy_samples,
            normalize_mean=self._mean,
            normalize_std=self._std,
            filter_observed_data=self._filter_observed_data,
            hp_ranges_for_prediction=self._hp_ranges_for_prediction,
        )

    def _get_num_fantasy_samples(self) -> int:
        raise NotImplementedError()

    def _posterior_for_state(
        self,
        state: TuningJobState,
        fit_params: bool,
        profiler: Optional[SimpleProfiler] = None,
    ):
        """
        Computes posterior for state.
        If fit_params and state.pending_evaluations is empty, we first
        optimize the model parameters.
        If state.pending_evaluations are given, these must be
        FantasizedPendingEvaluations, i.e. the fantasy values must have been
        sampled.
        """
        assert state.num_observed_cases(self.active_metric) > 0, (
            "Cannot compute posterior: state has no labeled datapoints "
            + f"for metric {self.active_metric}"
        )
        internal_candidate_evaluations = get_internal_candidate_evaluations(
            state,
            self.active_metric,
            self.normalize_targets,
            self._get_num_fantasy_samples(),
        )
        features = internal_candidate_evaluations.features
        targets = internal_candidate_evaluations.targets
        assert features.shape[0] == targets.shape[0]
        self._mean = internal_candidate_evaluations.mean
        self._std = internal_candidate_evaluations.std

        fit_params = fit_params and (not state.pending_evaluations)
        data = {"features": features, "targets": targets}
        if not fit_params:
            if self._debug_log is not None:
                logger.info("Recomputing posterior state")
            self._gpmodel.recompute_states(data)
        else:
            if self._debug_log is not None:
                logger.info(f"Fitting surrogate model for {self.active_metric}")
            self._gpmodel.fit(data, profiler=profiler)
        if self._debug_log is not None:
            self._debug_log.set_model_params(self.get_params())
            if not state.pending_evaluations:
                deb_msg = "[GaussProcModelFactory._posterior_for_state]\n"
                deb_msg += "- self.mean = {}\n".format(self._mean)
                deb_msg += "- self.std = {}".format(self._std)
                logger.info(deb_msg)
                self._debug_log.set_targets(targets)
            else:
                num_pending = len(state.pending_evaluations)
                fantasies = targets[-num_pending:, :]
                self._debug_log.set_fantasies(fantasies)

    def _num_samples_for_fantasies(self) -> int:
        raise NotImplementedError()

    def _draw_fantasy_values(self, state: TuningJobState) -> TuningJobState:
        """
        Note: The fantasy values need not be de-normalized, because they are
        only used internally here (e.g., get_internal_candidate_evaluations).

        Note: A complication is that if the sampling methods of _gpmodel
        are called when there are no pending candidates (with fantasies) yet,
        they do return a single sample (instead of num_fantasy_samples). This
        is because GaussianProcessRegression knows about num_fantasy_samples
        only due to the form of the posterior state (bad design!).
        In this case, we draw num_fantasy_samples i.i.d.

        """
        if state.pending_evaluations:
            configs = state.pending_configurations()
            features_new = state.hp_ranges.to_ndarray_matrix(configs)
            num_samples = self._num_samples_for_fantasies()
            # We need joint sampling for >1 new candidates
            num_candidates = len(configs)
            sample_func = (
                self._gpmodel.sample_joint
                if num_candidates > 1
                else self._gpmodel.sample_marginals
            )
            targets_new = sample_func(
                features_test=features_new, num_samples=num_samples
            ).reshape((num_candidates, -1))
            new_pending = [
                FantasizedPendingEvaluation(
                    trial_id=ev.trial_id,
                    resource=ev.resource,
                    fantasies={self.active_metric: y_new.reshape((1, -1))},
                )
                for ev, y_new in zip(state.pending_evaluations, targets_new)
            ]
        else:
            new_pending = []
        return TuningJobState(
            hp_ranges=state.hp_ranges,
            config_for_trial=state.config_for_trial,
            trials_evaluations=state.trials_evaluations,
            failed_trials=state.failed_trials,
            pending_evaluations=new_pending,
        )

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler

        if isinstance(
            self._gpmodel, (IndependentGPPerResourceModel, HyperTuneJointGPModel)
        ):
            assert isinstance(scheduler, HyperbandScheduler), (
                "gpmodel of type IndependentGPPerResourceModel requires "
                + "HyperbandScheduler scheduler"
            )
            # Likelihood of internal model still has to be created (depends on
            # rung levels of scheduler). Note that `max_t` must be included
            max_t = scheduler.max_t
            if scheduler.rung_levels[-1] == max_t:
                rung_levels = scheduler.rung_levels
            else:
                rung_levels = scheduler.rung_levels + [max_t]
            self._gpmodel.create_likelihood(rung_levels)


class GaussProcEmpiricalBayesModelFactory(GaussProcModelFactory):
    def __init__(
        self,
        gpmodel: GPModel,
        num_fantasy_samples: int,
        active_metric: str = INTERNAL_METRIC_NAME,
        normalize_targets: bool = True,
        profiler: Optional[SimpleProfiler] = None,
        debug_log: Optional[DebugLogPrinter] = None,
        filter_observed_data: Optional[ConfigurationFilter] = None,
        no_fantasizing: bool = False,
        hp_ranges_for_prediction: Optional[HyperparameterRanges] = None,
    ):
        """
        We support pending evaluations via fantasizing. Note that state does
        not contain the fantasy values, but just the pending configs. Fantasy
        values are sampled here.

        :param gpmodel: GaussianProcessRegression model
        :param num_fantasy_samples: See above
        :param active_metric: Name of the metric to optimize.
        :param normalize_targets: Normalize target values in
            state.candidate_evaluations?

        """
        assert num_fantasy_samples > 0
        super().__init__(
            gpmodel=gpmodel,
            active_metric=active_metric,
            normalize_targets=normalize_targets,
            profiler=profiler,
            debug_log=debug_log,
            filter_observed_data=filter_observed_data,
            no_fantasizing=no_fantasizing,
            hp_ranges_for_prediction=hp_ranges_for_prediction,
        )
        self.num_fantasy_samples = num_fantasy_samples

    def get_params(self):
        return self._gpmodel.get_params()

    def set_params(self, param_dict):
        self._gpmodel.set_params(param_dict)

    def _get_num_fantasy_samples(self) -> int:
        return self.num_fantasy_samples

    def _num_samples_for_fantasies(self) -> int:
        # Special case (see header comment): If the current posterior state
        # does not contain pending candidates (no fantasies), we sample
        # `num_fantasy_samples` times i.i.d.
        return 1 if self._gpmodel.multiple_targets() else self.num_fantasy_samples
