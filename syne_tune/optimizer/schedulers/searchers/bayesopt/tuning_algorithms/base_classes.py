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
from typing import (
    List,
    Iterable,
    Tuple,
    Type,
    Optional,
    Set,
    Dict,
    Union,
    Any,
    Iterator,
)
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import ExclusionList
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)


def assign_active_metric(model, active_metric):
    """Checks that active_metric is provided when model consists of multiple output models.
    Otherwise, just sets active_metric to the only model output name available.
    """
    model_output_names = sorted(model.keys())
    num_output_models = len(model_output_names)
    if num_output_models == 1:
        if active_metric is not None:
            assert active_metric == model_output_names[0], (
                "Only a single output model is given. "
                "Active metric must be set to that output model."
            )
        active_metric = model_output_names[0]
    else:
        assert active_metric is not None, (
            f"As model has {num_output_models}, active metric cannot be None. "
            f"Please set active_metric to one of the model output names: "
            f"{model_output_names}."
        )
        assert active_metric in model_output_names
    return active_metric


class NextCandidatesAlgorithm:
    def next_candidates(self) -> List[Configuration]:
        raise NotImplementedError


class SurrogateModel:
    """
    Base class of surrogate models for Bayesian optimization. A surrogate model
    supports marginal predictions feeding into an acquisition function, as well
    as suppport to compute gradients of an acquisition function w.r.t. inputs.

    :param state: Tuning job state
    :param active_metric: Name of internal objective
    """

    def __init__(self, state: TuningJobState, active_metric: Optional[str] = None):
        self.state = state
        if active_metric is None:
            active_metric = INTERNAL_METRIC_NAME
        self.active_metric = active_metric

    @staticmethod
    def keys_predict() -> Set[str]:
        """Keys of signals returned by :meth:`predict`.

        Note: In order to work with :class:`AcquisitionFunction` implementations,
        the following signals are required:

        * "mean": Predictive mean
        * "std": Predictive standard deviation

        :return: Set of keys for ``dict`` returned by :meth:`predict`
        """
        return {"mean", "std"}

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Returns signals which are statistics of the predictive distribution at
        input points ``inputs``. By default:

        * "mean": Predictive means. If the model supports fantasizing with a
            number ``nf`` of fantasies, this has shape ``(n, nf)``, otherwise ``(n,)``
        - "std": Predictive stddevs, shape ``(n,)``

        If the hyperparameters of the surrogate model are being optimized (e.g.,
        by empirical Bayes), the returned list has length 1. If its
        hyperparameters are averaged over by MCMC, the returned list has one
        entry per MCMC sample.

        :param inputs: Input points, shape ``(n, d)``
        :return: List of ``dict`` with keys :meth:`keys_predict`, of length the
            number of MCMC samples, or length 1 for empirical Bayes
        """
        raise NotImplementedError

    def hp_ranges_for_prediction(self) -> HyperparameterRanges:
        """
        :return: Feature generator to be used for ``inputs`` in :meth:`predict`
        """
        return self.state.hp_ranges

    def predict_candidates(
        self, candidates: Iterable[Configuration]
    ) -> List[Dict[str, np.ndarray]]:
        """Convenience variant of :meth:`predict`

        :param candidates: List of configurations
        :return: Same as :meth:`predict`
        """
        return self.predict(
            self.hp_ranges_for_prediction().to_ndarray_matrix(candidates)
        )

    def current_best(self) -> List[np.ndarray]:
        """
        Returns the so-called incumbent, to be used in acquisition functions
        such as expected improvement. This is the minimum of predictive means
        (signal with key "mean") at all current candidate locations (both
        state.trials_evaluations and state.pending_evaluations).
        Normally, a scalar is returned, but if the model supports fantasizing
        and the state contains pending evaluations, there is one incumbent
        per fantasy sample, so a vector is returned.

        If the hyperparameters of the surrogate model are being optimized (e.g.,
        by empirical Bayes), the returned list has length 1. If its
        hyperparameters are averaged over by MCMC, the returned list has one
        entry per MCMC sample.

        :return: Incumbent, see above
        """
        raise NotImplementedError

    def backward_gradient(
        self, input: np.ndarray, head_gradients: List[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Computes the gradient :math:`\nabla f(x)` for an acquisition
        function :math:`f(x)`, where :math:`x` is a single input point. This
        is using reverse mode differentiation, the head gradients are passed
        by the acquisition function. The head gradients are
        :math:`\partial_k f`, where :math:`k` runs over the statistics
        returned by :meth:`predict` for the single input point :math:`x`.
        The shape of head gradients is the same as the shape of the
        statistics.

        Lists have ``> 1`` entry if MCMC is used, otherwise they are all size 1.

        :param input: Single input point :math:`x`, shape ``(d,)``
        :param head_gradients: See above
        :return: Gradient :math:`\nabla f(x)` (several if MCMC is used)
        """
        raise NotImplementedError


# Useful type that allows for a dictionary mapping each output name to a SurrogateModel.
# This is needed for multi-output BO methods such as constrained BO, where each output
# is associated to a model. This type includes the Union with the standard
# SurrogateModel type for backward compatibility.
SurrogateOutputModel = Union[SurrogateModel, Dict[str, SurrogateModel]]


class ScoringFunction:
    """
    Class to score candidates. As opposed to acquisition functions, scores do
    not support gradient computation. Note that scores are always minimized.
    """

    def score(
        self,
        candidates: Iterable[Configuration],
        model: Optional[SurrogateOutputModel] = None,
    ) -> List[float]:
        """
        :param candidates: Configurations for which scores are to be computed
        :param model: Overrides default surrogate model
        :return: List of score values, length of ``candidates``
        """
        raise NotImplementedError


class AcquisitionFunction(ScoringFunction):
    """
    Base class for acquisition functions :math:`f(x)`.

    :param model: Surrogate model providing predictive statistics (e.g.,
        mean, variance)
    :param active_metric: Name of internal metric
    """

    def __init__(
        self, model: SurrogateOutputModel, active_metric: Optional[str] = None
    ):
        self.model = model
        if active_metric is None:
            active_metric = INTERNAL_METRIC_NAME
        self.active_metric = active_metric

    def compute_acq(
        self, inputs: np.ndarray, model: Optional[SurrogateOutputModel] = None
    ) -> np.ndarray:
        """
        Note: If inputs has shape ``(d,)``, it is taken to be ``(1, d)``

        :param inputs: Encoded input points, shape ``(n, d)``
        :param model: If given, overrides ``self.model``
        :return: Acquisition function values, shape ``(n,)``
        """
        raise NotImplementedError

    def compute_acq_with_gradient(
        self, input: np.ndarray, model: Optional[SurrogateOutputModel] = None
    ) -> Tuple[float, np.ndarray]:
        """
        For a single input point :math:`x`, compute acquisition function value
        :math:`f(x)` and gradient :math:`\nabla f(x)`.

        :param input: Single input point :math:`x`, shape ``(d,)``
        :param model: If given, overrides ``self.model``
        :return: :math:`(f(x), \nabla f(x))`
        """
        raise NotImplementedError

    def score(
        self,
        candidates: Iterable[Configuration],
        model: Optional[SurrogateOutputModel] = None,
    ) -> List[float]:
        if model is None:
            model = self.model
        if isinstance(model, dict):
            active_model = model[self.active_metric]
        else:
            active_model = model
        hp_ranges = active_model.hp_ranges_for_prediction()
        inputs = hp_ranges.to_ndarray_matrix(candidates)
        return list(self.compute_acq(inputs, model=model))


AcquisitionClassAndArgs = Union[
    Type[AcquisitionFunction], Tuple[Type[AcquisitionFunction], Dict[str, Any]]
]


def unwrap_acquisition_class_and_kwargs(
    acquisition_class: AcquisitionClassAndArgs,
) -> (Type[AcquisitionFunction], Dict[str, Any]):
    if isinstance(acquisition_class, tuple):
        return acquisition_class
    else:
        return acquisition_class, dict()


class LocalOptimizer:
    """
    Class that tries to find a local candidate with a better score, typically
    using a local optimization method such as L-BFGS. It would normally
    encapsulate an acquisition function and model.

    ``acquisition_class`` contains the type of the acquisition function
    (subclass of :class:`AcquisitionFunction`). It can also be a tuple of the
    form ``(type, kwargs)``, where ``kwargs`` are extra arguments to the class
    constructor.

    :param hp_ranges: Feature generator for configurations
    :param model: Surrogate model for acquisition function
    :param acquisition_class: See above
    :param active_metric: Name of internal metric
    """

    def __init__(
        self,
        hp_ranges: HyperparameterRanges,
        model: SurrogateOutputModel,
        acquisition_class: AcquisitionClassAndArgs,
        active_metric: Optional[str] = None,
    ):
        self.hp_ranges = hp_ranges
        self.model = model
        if active_metric is None:
            active_metric = INTERNAL_METRIC_NAME
        if isinstance(model, dict):
            self.active_metric = assign_active_metric(model, active_metric)
        else:
            self.active_metric = active_metric
        self.acquisition_class = acquisition_class

    def optimize(
        self, candidate: Configuration, model: Optional[SurrogateOutputModel] = None
    ) -> Configuration:
        """Run local optimization, starting from ``candidate``

        :param candidate: Starting point
        :param model: Overrides ``self.model``
        :return: Configuration found by local optimization
        """
        raise NotImplementedError


class CandidateGenerator:
    """
    Class to generate candidates from which to start the local minimization,
    typically random candidate or some form of more uniformly spaced variation,
    such as latin hypercube or Sobol sequence.
    """

    def generate_candidates(self) -> Iterator[Configuration]:
        raise NotImplementedError

    def generate_candidates_en_bulk(
        self, num_cands: int, exclusion_list: Optional[ExclusionList] = None
    ) -> List[Configuration]:
        """
        :param num_cands: Number of candidates to generate
        :param exclusion_list: If given, these candidates must not be returned
        :return: List of ``num_cands`` candidates. If ``exclusion_list`` is given,
            the number of candidates returned can be ``< num_cands``
        """
        raise NotImplementedError
