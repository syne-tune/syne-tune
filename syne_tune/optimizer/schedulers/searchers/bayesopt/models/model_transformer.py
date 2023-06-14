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
from typing import Dict, Optional, Callable, Union
import logging
import copy

from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator import (
    Estimator,
    OutputEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    OutputPredictor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_skipopt import (
    SkipOptimizationPredicate,
    NeverSkipPredicate,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    PendingEvaluation,
    TrialEvaluations,
    dictionarize_objective,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration

logger = logging.getLogger(__name__)


def _assert_same_keys(dict1, dict2):
    assert set(dict1.keys()) == set(
        dict2.keys()
    ), f"{list(dict1.keys())} and {list(dict2.keys())} need to be the same keys. "


# Convenience type allowing for multi-output HPO. These are used for methods
# that work both in the standard case of a single output model and in the
# multi-output case

SkipOptimizationOutputPredicate = Union[
    SkipOptimizationPredicate, Dict[str, SkipOptimizationPredicate]
]


class StateForModelConverter:
    """
    Interface for state converters (optionally) used in
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer.ModelStateTransformer`.
    These are applied to a state before being passed to the model for fitting and
    predictions. The main use case is to filter down data if fitting the model scales
    super-linearly.
    """

    def __call__(self, state: TuningJobState) -> TuningJobState:
        raise NotImplementedError

    def set_random_state(self, random_state: RandomState):
        """
        Some state converters use random sampling. For these, the random state has to
        be set before first usage.

        :param random_state: Random state to be used internally
        """
        pass


class ModelStateTransformer:
    """
    This class maintains the
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state.TuningJobState`
    object alongside an HPO experiment, and manages the reaction to changes of
    this state. In particular, it provides a fitted surrogate model on demand,
    which encapsulates the GP posterior.

    The state transformer is generic, it uses :class:`Estimator` for anything specific
    to the model type.

    ``skip_optimization`` is a predicate depending on the state, determining
    what is done at the next recent call of ``model``. If ``False``, the model
    parameters are refit, otherwise the current ones are not changed (which
    is usually faster, but risks stale-ness).

    We also track the observed data ``state.trials_evaluations``. If this
    did not change since the last recent :meth:`model` call, we do not refit the
    model parameters. This is based on the assumption that model parameter
    fitting only depends on ``state.trials_evaluations`` (observed data),
    not on other fields (e.g., pending evaluations).

    If given, ``state_converter`` maps the state to another one which is then
    passed to the model for fitting and predictions. One important use case is
    filtering down data when model fitting is superlinear. Another is to convert
    multi-fidelity setups to be used with single-fidelity models inside.

    Note that ``estimator`` and ``skip_optimization`` can also be a dictionary mapping
    output names to models. In that case, the state is shared but the models for each
    output metric are updated independently.

    :param estimator: Surrogate model(s)
    :param init_state: Initial tuning job state
    :param skip_optimization: Skip optimization predicate (see above). Defaults to
        ``None`` (fitting is never skipped)
    :param state_converter: See above, optional
    """

    def __init__(
        self,
        estimator: OutputEstimator,
        init_state: TuningJobState,
        skip_optimization: Optional[SkipOptimizationOutputPredicate] = None,
        state_converter: Optional[StateForModelConverter] = None,
    ):
        self._use_single_model = False
        if isinstance(estimator, Estimator):
            self._use_single_model = True
        if not self._use_single_model:
            assert isinstance(estimator, dict), (
                f"{estimator} is not an instance of Estimator. "
                f"It is assumed that we are in the multi-output case and that it "
                f"must be a Dict. No other types are supported. "
            )
            # Default: Always refit model parameters for each output model
            if skip_optimization is None:
                skip_optimization = {
                    output_name: NeverSkipPredicate()
                    for output_name in estimator.keys()
                }
            else:
                assert isinstance(skip_optimization, Dict), (
                    f"{skip_optimization} must be a Dict, consistently "
                    f"with {estimator}."
                )
                _assert_same_keys(estimator, skip_optimization)
                skip_optimization = {
                    output_name: skip_optimization[output_name]
                    if skip_optimization.get(output_name) is not None
                    else NeverSkipPredicate()
                    for output_name in estimator.keys()
                }
            # debug_log is shared by all output models
            self._debug_log = next(iter(estimator.values())).debug_log
        else:
            if skip_optimization is None:
                # Default: Always refit model parameters
                skip_optimization = NeverSkipPredicate()
            assert isinstance(skip_optimization, SkipOptimizationPredicate)
            self._debug_log = estimator.debug_log
            # Make ``estimator`` and ``skip_optimization`` single-key dictionaries
            # for convenience, so that we can treat the single model and multi-model case in the same way
            estimator = dictionarize_objective(estimator)
            skip_optimization = dictionarize_objective(skip_optimization)
        self._estimator = estimator
        self._skip_optimization = skip_optimization
        self._state_converter = state_converter
        self._state = copy.copy(init_state)
        # OutputPredictor computed on demand
        self._predictor: Optional[OutputPredictor] = None
        # Observed data for which model parameters were re-fit most
        # recently, separately for each model
        self._num_evaluations = {output_name: 0 for output_name in estimator.keys()}

    @property
    def state(self) -> TuningJobState:
        return self._state

    def _unwrap_from_dict(self, x):
        if self._use_single_model:
            return next(iter(x.values()))
        else:
            return x

    @property
    def use_single_model(self) -> bool:
        return self._use_single_model

    @property
    def estimator(self) -> OutputEstimator:
        return self._unwrap_from_dict(self._estimator)

    @property
    def skip_optimization(self) -> SkipOptimizationOutputPredicate:
        return self._unwrap_from_dict(self._skip_optimization)

    def fit(self, **kwargs) -> OutputPredictor:
        """
        If ``skip_optimization`` is given, it overrides the ``self._skip_optimization``
        predicate.

        :return: Fitted surrogate model for current state in the standard single
            model case; in the multi-model case, it returns a dictionary mapping
            output names to surrogate model instances for current state (shared
            across models).
        """
        if self._predictor is None:
            skip_optimization = kwargs.get("skip_optimization")
            self._compute_predictor(skip_optimization=skip_optimization)
        return self._unwrap_from_dict(self._predictor)

    def get_params(self):
        params = {
            output_name: output_estimator.get_params()
            for output_name, output_estimator in self._estimator.items()
        }
        return self._unwrap_from_dict(params)

    def set_params(self, param_dict):
        if self._use_single_model:
            param_dict = dictionarize_objective(param_dict)
        _assert_same_keys(self._estimator, param_dict)
        for output_name in self._estimator:
            self._estimator[output_name].set_params(param_dict[output_name])

    def append_trial(
        self,
        trial_id: str,
        config: Optional[Configuration] = None,
        resource: Optional[int] = None,
    ):
        """
        Appends new pending evaluation to the state.

        :param trial_id: ID of trial
        :param config: Must be given if this trial does not yet feature in the
            state
        :param resource: Must be given in the multi-fidelity case, to specify
            at which resource level the evaluation is pending
        """
        self._predictor = None  # Invalidate
        self._state.append_pending(trial_id, config=config, resource=resource)

    def drop_pending_evaluation(
        self, trial_id: str, resource: Optional[int] = None
    ) -> bool:
        """
        Drop pending evaluation from state. If it is not listed as pending,
        nothing is done

        :param trial_id: ID of trial
        :param resource: Must be given in the multi-fidelity case, to specify
            at which resource level the evaluation is pending

        """
        return self._state.remove_pending(trial_id, resource)

    def remove_observed_case(
        self,
        trial_id: str,
        metric_name: str = INTERNAL_METRIC_NAME,
        key: Optional[str] = None,
    ):
        """
        Removes specific observation from the state.

        :param trial_id: ID of trial
        :param metric_name: Name of internal metric
        :param key: Must be given in the multi-fidelity case
        """
        pos = self._state._find_labeled(trial_id)
        assert pos != -1, f"Trial trial_id = {trial_id} has no observations"
        metrics = self._state.trials_evaluations[pos].metrics
        assert metric_name in metrics, (
            f"state.trials_evaluations entry for trial_id = {trial_id} "
            + f"does not contain metric {metric_name}"
        )
        if key is None:
            del metrics[metric_name]
        else:
            metric_vals = metrics[metric_name]
            assert isinstance(metric_vals, dict) and key in metric_vals, (
                f"state.trials_evaluations entry for trial_id = {trial_id} "
                + f"and metric {metric_name} does not contain case for "
                + f"key {key}"
            )
            del metric_vals[key]

    def label_trial(
        self, data: TrialEvaluations, config: Optional[Configuration] = None
    ):
        """
        Adds observed data for a trial. If it has observations in the state
        already, ``data.metrics`` are appended. Otherwise, a new entry is
        appended.
        If new observations replace pending evaluations, these are removed.

        ``config`` must be passed if the trial has not yet been registered in
        the state (this happens normally with the ``append_trial`` call). If
        already registered, ``config`` is ignored.
        """
        # Drop pending candidate if it exists
        trial_id = data.trial_id
        metric_vals = data.metrics.get(INTERNAL_METRIC_NAME)
        if metric_vals is not None:
            resource_attr_name = self._state.hp_ranges.name_last_pos
            if resource_attr_name is not None:
                assert isinstance(
                    metric_vals, dict
                ), f"Metric {INTERNAL_METRIC_NAME} must be dict-valued"
                for resource in metric_vals.keys():
                    self.drop_pending_evaluation(trial_id, resource=int(resource))
            else:
                self.drop_pending_evaluation(trial_id)
        # Assign / append new labels
        metrics = self._state.metrics_for_trial(trial_id, config=config)
        for name, new_labels in data.metrics.items():
            if name not in metrics or not isinstance(new_labels, dict):
                metrics[name] = new_labels
            else:
                metrics[name].update(new_labels)
        self._predictor = None  # Invalidate

    def filter_pending_evaluations(
        self, filter_pred: Callable[[PendingEvaluation], bool]
    ):
        """
        Filters ``state.pending_evaluations`` with ``filter_pred``.

        :param filter_pred: Filtering predicate
        """
        new_pending_evaluations = list(
            filter(filter_pred, self._state.pending_evaluations)
        )
        if len(new_pending_evaluations) != len(self._state.pending_evaluations):
            self._predictor = None  # Invalidate
            del self._state.pending_evaluations[:]
            self._state.pending_evaluations.extend(new_pending_evaluations)

    def mark_trial_failed(self, trial_id: str):
        failed_trials = self._state.failed_trials
        if trial_id not in failed_trials:
            failed_trials.append(trial_id)

    def _compute_predictor(self, skip_optimization=None):
        if skip_optimization is None:
            skip_optimization = dict()
            for (
                output_name,
                output_skip_optimization,
            ) in self._skip_optimization.items():
                skip_optimization[output_name] = output_skip_optimization(self._state)
        elif self._use_single_model:
            skip_optimization = dictionarize_objective(skip_optimization)
        if self._debug_log is not None:
            for output_name, skip_opt in skip_optimization.items():
                if skip_opt:
                    logger.info(
                        f"Skipping the refitting of model parameters for {output_name}"
                    )

        _assert_same_keys(skip_optimization, self._estimator)
        state_for_model = (
            self._state
            if self._state_converter is None
            else self._state_converter(self._state)
        )
        output_predictors = dict()
        for output_name, output_skip_optimization in skip_optimization.items():
            update_params = not output_skip_optimization
            if update_params:
                # Did the labeled data really change since the last recent refit?
                # If not, skip the refitting
                num_evaluations = self._state.num_observed_cases(output_name)
                if num_evaluations == self._num_evaluations[output_name]:
                    update_params = False
                    if self._debug_log is not None:
                        logger.info(
                            f"Skipping the refitting of model parameters for {output_name}, "
                            f"since the labeled data did not change since the last recent fit"
                        )
                else:
                    # Model will be refitted: Update
                    self._num_evaluations[output_name] = num_evaluations
            output_predictors[output_name] = self._estimator[
                output_name
            ].fit_from_state(state=state_for_model, update_params=update_params)
        self._predictor = output_predictors
