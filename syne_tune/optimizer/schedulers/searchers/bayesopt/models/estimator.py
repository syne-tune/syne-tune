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
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    FantasizedPendingEvaluation,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    Predictor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)


class Estimator:
    """
    Interface for surrogate models used in :class:`ModelStateTransformer`.

    In general, a surrogate model is probabilistic (or Bayesian), in that
    predictions are driven by a posterior distribution, represented in a
    posterior state of type
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`.
    The model may also come with tunable (hyper)parameters, such as for example
    covariance function parameters for a Gaussian process model. These parameters
    can be accessed with :meth:`get_params`, :meth:`set_params`.
    """

    def get_params(self) -> Dict[str, Any]:
        """
        :return: Current tunable model parameters
        """
        raise NotImplementedError()

    def set_params(self, param_dict: Dict[str, Any]):
        """
        :param param_dict: New model parameters
        """
        raise NotImplementedError()

    def fit_from_state(self, state: TuningJobState, update_params: bool) -> Predictor:
        """
        Creates a
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`
        object based on data in ``state``. For a Bayesian model, this involves
        computing the posterior state, which is wrapped in the ``Predictor``
        object.

        If the model also has (hyper)parameters, these are learned iff
        ``update_params == True``. Otherwise, these parameters are not changed,
        but only the posterior state is computed. The idea is that in general,
        model fitting is much more expensive than just creating the final posterior
        state (or predictor). It then makes sense to partly work with stale model
        parameters.

        If your surrogate model is not Bayesian, or does not have hyperparameters,
        you can ignore the ``update_params`` argument,

        :param state: Current data model parameters are to be fit on, and the
            posterior state is to be computed from
        :param update_params: See above
        :return: Predictor, wrapping the posterior state
        """
        raise NotImplementedError()

    @property
    def debug_log(self) -> Optional[DebugLogPrinter]:
        return None

    def configure_scheduler(self, scheduler):
        """
        Called by :meth:`configure_scheduler` of searchers which make use of an
        class:``Estimator``. Allows the estimator to depend on
        parameters of the scheduler.

        :param scheduler: Scheduler object
        """
        pass


# Convenience type allowing for multi-output HPO. These are used for methods
# that work both in the standard case of a single output model and in the
# multi-output case

OutputEstimator = Union[Estimator, Dict[str, Estimator]]


@dataclass
class TransformedData:
    features: np.ndarray
    targets: np.ndarray
    mean: float
    std: float


def transform_state_to_data(
    state: TuningJobState,
    active_metric: Optional[str] = None,
    normalize_targets: bool = True,
    num_fantasy_samples: int = 1,
) -> TransformedData:
    """
    Transforms
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state.TuningJobState`
    object ``state`` to features and targets. The former are encoded vectors from
    ``state.hp_ranges``. The latter are normalized to zero mean, unit variance if
    ``normalize_targets == True``, in which case the original mean and stddev is
    also returned.

    If ``state.pending_evaluations`` is not empty, it must contain entries
    of type
    :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common.FantasizedPendingEvaluation`,
    which contain the fantasy samples. This is the case only for internal states.

    :param state: ``TuningJobState`` to transform
    :param active_metric: Name of target metric (optional)
    :param normalize_targets: Normalize targets? Defaults to ``True``
    :param num_fantasy_samples: Number of fantasy samples. Defaults to 1
    :return: Transformed data
    """
    if active_metric is None:
        active_metric = INTERNAL_METRIC_NAME
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
    return TransformedData(features, targets, mean, std)
