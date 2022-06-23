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
import numpy as np
from typing import Dict, Tuple, Optional, Set, List
from dataclasses import dataclass
import itertools

from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    SurrogateModel,
    AcquisitionFunction,
    SurrogateOutputModel,
    assign_active_metric,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    dictionarize_objective,
)


# Type for predictions from (potentially) multiple models
# `output_to_predictions[name]` is a list of dicts, one entry for each
# MCMC sample (list is size 1 if no MCMC), see also `predict` of
# :class:`SurrogateModel`.
# Note: List sizes of different entries can be different. MCMC averaging
# is done over the Cartesian product of these lists.
PredictionsPerOutput = Dict[str, List[Dict[str, np.ndarray]]]

SamplePredictionsPerOutput = Dict[str, Dict[str, np.ndarray]]


@dataclass
class HeadWithGradient:
    """
    `gradient` maps each output model to a dict of head gradients, whose keys
    are those used by `predict` (e.g., `mean`, `std`)
    """

    hval: np.ndarray
    gradient: SamplePredictionsPerOutput


class CurrentBestProvider:
    """
    Helper class for :class:`MeanStdAcquisitionFunction`.
    The `current_best` values required in `compute_acq` and
    `compute_acq_with_gradient` may depend on the MCMC sample index for each
    model (if none of the models use MCMC, this index is always
    (0, 0, ..., 0)).

    """

    def __call__(self, positions: Tuple[int, ...]) -> Optional[np.ndarray]:
        raise NotImplementedError


class NoneCurrentBestProvider(CurrentBestProvider):
    def __call__(self, positions: Tuple[int, ...]) -> Optional[np.ndarray]:
        return None


class ActiveMetricCurrentBestProvider(CurrentBestProvider):
    """
    Default implementation in which `current_best` depends on the
    active metric only.

    """

    def __init__(self, active_metric_current_best: List[np.ndarray]):
        self._active_metric_current_best = [
            v.reshape((1, -1)) for v in active_metric_current_best
        ]
        self._constant_list = len(active_metric_current_best) == 1

    def __call__(self, positions: Tuple[int, ...]) -> Optional[np.ndarray]:
        pos = positions[0] if not self._constant_list else 0
        return self._active_metric_current_best[pos]


class MeanStdAcquisitionFunction(AcquisitionFunction):
    """
    Base class for standard acquisition functions which depend on predictive
    mean and stddev. Subclasses have to implement the head and its derivatives
    w.r.t. mean and std:

        f(x, model) = h(mean, std, model.current_best())

    If model is a SurrogateModel, then active_metric is ignored. If model is a Dict mapping output names to models,
    then active_metric must be given.

    NOTE that acquisition functions will always be *minimized*!

    """

    def __init__(self, model: SurrogateOutputModel, active_metric: str = None):
        super().__init__(model, active_metric)
        if isinstance(model, SurrogateModel):
            # Ignore active_metric
            model = dictionarize_objective(model)
        assert isinstance(model, Dict)
        self.model = model
        self.model_output_names = sorted(model.keys())
        self.active_metric = assign_active_metric(model, active_metric)
        output_names = list(model.keys())
        active_pos = output_names.index(self.active_metric)
        # active_metric to come first
        self.model_output_names = (
            [self.active_metric]
            + output_names[:active_pos]
            + output_names[(active_pos + 1) :]
        )
        self._check_keys_predict_of_models()
        self._current_bests = None

    def _output_to_keys_predict(self) -> Dict[str, Set[str]]:
        """
        Required `keys_predict` for each output model. The default requires
        each output model to return 'mean' and 'std'.

        """
        mean_and_std = {"mean", "std"}
        return {k: mean_and_std for k in self.model_output_names}

    def _check_keys_predict_of_models(self):
        for output_name, required_keys in self._output_to_keys_predict().items():
            keys_predict = self.model[output_name].keys_predict()
            for k in required_keys:
                assert k in keys_predict, (
                    f"output_name {output_name}: Required key {k} not "
                    + "provided by predictions of surrogate model"
                )

    def _get_num_fantasies(self, output_to_predictions: PredictionsPerOutput) -> int:
        """
        If fantasizing is used, the number of fantasy samples must be
        the same over all models. Even if this number is >1, a model
        may always not use fantasizing, in which case its mean predictions
        are broadcasted.

        :param output_to_predictions:
        :return: Number of fantasies
        """
        num_fantasy_values = set()
        for predictions in output_to_predictions.values():
            for prediction in predictions:
                assert "mean" in prediction  # Sanity check
                means = prediction["mean"]
                num_fantasies = means.shape[1] if means.ndim == 2 else 1
                num_fantasy_values.add(num_fantasies)
        max_size = 2 if (1 in num_fantasy_values) else 1
        assert (
            len(num_fantasy_values) <= max_size
        ), "Predictive means have inconsistent numbers of fantasies: " + str(
            num_fantasy_values
        )
        return max(list(num_fantasy_values))

    def _get_current_bests(self, model: SurrogateOutputModel) -> CurrentBestProvider:
        current_bests = self._current_bests
        default_model = model is self.model
        if (not default_model) or current_bests is None:
            if self._head_needs_current_best():
                current_bests = self._get_current_bests_internal(model)
            else:
                current_bests = NoneCurrentBestProvider()
            if default_model:
                self._current_bests = current_bests
        return current_bests

    def _get_current_bests_internal(
        self, model: SurrogateOutputModel
    ) -> CurrentBestProvider:
        """
        Implements default where `current_best` only depends on the model for
        `active_metric`. To be overwritten by child classes where this does not
        hold.

        Note: The resulting current_bests is redetermined every time, since
        `model` may change.

        """
        active_metric_current_best = model[self.active_metric].current_best()
        return ActiveMetricCurrentBestProvider(active_metric_current_best)

    def compute_acq(
        self, inputs: np.ndarray, model: Optional[SurrogateOutputModel] = None
    ) -> np.ndarray:
        if model is None:
            model = self.model
        elif isinstance(model, SurrogateModel):
            model = dictionarize_objective(model)
        if inputs.ndim == 1:
            inputs = inputs.reshape((1, -1))
        output_to_predictions = self._map_outputs_to_predictions(model, inputs)
        current_bests = self._get_current_bests(model)

        # Reshaping of predictions to accomodate _compute_head.
        for preds_for_samples in output_to_predictions.values():
            for prediction in preds_for_samples:
                for k in prediction.keys():
                    v = prediction[k]
                    if (k == "mean" and v.ndim == 1) or k == "std":
                        prediction[k] = v.reshape((-1, 1))

        # MCMC average is product over lists coming from each model. The
        # resulting function values are stored in a flat list.
        fvals_list = []
        # We also need the position in each list in order to select
        # current_best
        list_values = [
            list(enumerate(output_to_predictions[name]))
            for name in self.model_output_names
        ]
        for preds_and_pos in itertools.product(*list_values):
            positions, predictions = zip(*preds_and_pos)
            output_to_preds = dict(zip(self.model_output_names, predictions))
            current_best = current_bests(positions)
            # Compute the acquisition function value
            fvals = self._compute_head(output_to_preds, current_best)
            fvals_list.append(fvals.reshape((-1,)))

        return np.mean(fvals_list, axis=0)

    @staticmethod
    def _add_head_gradients(
        grad1: Dict[str, np.ndarray], grad2: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if grad2 is None:
            return grad1
        else:
            return {k: v + grad2[k] for k, v in grad1.items()}

    def compute_acq_with_gradient(
        self, input: np.ndarray, model: Optional[SurrogateOutputModel] = None
    ) -> (float, np.ndarray):
        if model is None:
            model = self.model
        if isinstance(model, SurrogateModel):
            model = dictionarize_objective(model)
        output_to_predictions = self._map_outputs_to_predictions(
            model, input.reshape(1, -1)
        )
        current_bests = self._get_current_bests(model)

        # Reshaping of predictions to accomodate _compute_head_and_gradient. We
        # also store the original shapes, which are needed below
        shapes = dict()
        for output_name, preds_for_samples in output_to_predictions.items():
            shapes[output_name] = {k: v.shape for k, v in preds_for_samples[0].items()}
            for prediction in preds_for_samples:
                for k in prediction.keys():
                    prediction[k] = prediction[k].reshape((-1,))

        # MCMC average is product over lists coming from each model. We need to
        # accumulate head gradients w.r.t. each model, each of which being a
        # list over MCMC samples from that model (size 1 if no MCMC)
        fvals_list = []
        # For accumulation of head gradients, we also need to have the
        # position in each list
        list_values = [
            list(enumerate(output_to_predictions[name]))
            for name in self.model_output_names
        ]
        head_gradient = {
            name: [None] * len(predictions)
            for name, predictions in output_to_predictions.items()
        }
        for preds_and_pos in itertools.product(*list_values):
            positions, predictions = zip(*preds_and_pos)
            output_to_preds = dict(zip(self.model_output_names, predictions))
            current_best = current_bests(positions)
            head_result = self._compute_head_and_gradient(output_to_preds, current_best)
            fvals_list.append(head_result.hval)
            for output_name, pos in zip(self.model_output_names, positions):
                head_gradient[output_name][pos] = self._add_head_gradients(
                    head_result.gradient[output_name], head_gradient[output_name][pos]
                )

        # Sum up the gradients coming from each output model
        fval = np.mean(fvals_list)
        num_total = len(fvals_list)
        gradient = 0.0
        for output_name, output_model in model.items():
            # Reshape head gradients so they have the same shape as corresponding
            # predictions. This is required for `backward_gradient` to work.
            shp = shapes[output_name]
            head_grad = [
                {k: v.reshape(shp[k]) for k, v in orig_grad.items()}
                for orig_grad in head_gradient[output_name]
            ]
            # Gradients are computed by the model
            gradient_list = output_model.backward_gradient(input, head_grad)
            # Average over MCMC samples
            output_gradient = np.sum(gradient_list, axis=0) / num_total
            gradient += output_gradient
        return fval, gradient

    def _map_outputs_to_predictions(
        self, model: SurrogateOutputModel, inputs: np.ndarray
    ) -> PredictionsPerOutput:
        return {
            output_name: output_model.predict(inputs)
            for output_name, output_model in model.items()
        }

    def _extract_mean_and_std(
        self, output_to_predictions: SamplePredictionsPerOutput, metric: str = None
    ) -> (np.ndarray, np.ndarray):
        if metric is None:
            metric = self.active_metric
        predictions = output_to_predictions[metric]
        return predictions["mean"], predictions["std"]

    def _head_needs_current_best(self) -> bool:
        """
        :return: Is the current_best argument in _compute_head needed?
        """
        raise NotImplementedError

    def _compute_head(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        If mean has nf > 1 columns, both std and current_best are supposed to
        be broadcasted, and the return value is averaged over this dimension.

        :param output_to_predictions: Dictionary mapping each output to a
            dict containing predictive moments, keys as in
            `_output_to_keys_predict`. 'mean' has shape (n, nf), 'std' has
            shape (n, 1)
        :param current_best: Incumbent, shape (1, nf)
        :return: h(predictions, current_best), shape (n,)

        """
        raise NotImplementedError

    def _compute_head_and_gradient(
        self,
        output_to_predictions: SamplePredictionsPerOutput,
        current_best: Optional[np.ndarray],
    ) -> HeadWithGradient:
        """
        Computes both head value and head gradients, for a single input.

        :param: output_to_predictions: Dictionary mapping each output to a
            dict containing predictive moments, keys as in
            `_output_to_keys_predict`. 'mean' has shape (nf,), 'std' has shape
            (1,)
        :param current_best: Incumbent, shape (nf,)
        :return: HeadWithGradient containing hval and head gradients for
            each output model. All HeadWithGradient values have the same
            shape as the corresponding predictions

        """
        raise NotImplementedError
