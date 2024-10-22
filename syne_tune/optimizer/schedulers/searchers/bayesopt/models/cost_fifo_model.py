from typing import Dict, List, Set
import numpy as np
import logging

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator import Estimator
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_base import (
    BasePredictor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.cost.cost_model import (
    CostModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    Predictor,
)

logger = logging.getLogger(__name__)


class CostFixedResourcePredictor(BasePredictor):
    """
    Wraps cost model :math:`c(x, r)` of :class:`CostModel` to be used as
    surrogate model, where predictions are done at r = ``fixed_resource``.

    Note: For random cost models, we approximate expectations in ``predict``
    by resampling ``num_samples`` times (should be 1 for deterministic cost
    models).

    Note: Since this is a generic wrapper, we assume for ``backward_gradient``
    that the gradient contribution through the cost model vanishes. For special
    cost models, the mapping from encoded input to predictive means may be
    differentiable, and prediction code in ``autograd`` may be available. For
    such cost models, this wrapper should not be used, and ``backward_gradient``
    should be implemented properly.

    :param state: TuningJobSubState
    :param model: Model parameters must have been fit
    :param fixed_resource: :math:`c(x, r)` is predicted for this resource level r
    :param num_samples: Number of samples drawn in :meth:`predict`. Use this for
        random cost models only
    """

    def __init__(
        self,
        state: TuningJobState,
        model: CostModel,
        fixed_resource: int,
        num_samples: int = 1,
    ):
        super().__init__(state, active_metric=model.cost_metric_name)
        self._model = model
        self._fixed_resource = fixed_resource
        self._num_samples = num_samples

    @staticmethod
    def keys_predict() -> Set[str]:
        return {"mean"}

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        # Inputs are encoded. For cost models, need to decode them back
        # to candidates (not necessarily differentiable)
        # Note: Both ``inputs`` and ``hp_ranges`` may correspond to extended
        # configs (where one attribute is the resource level). This is not
        # a problem, since the resource attribute is simply ignored by the
        # cost model.
        hp_ranges = self.hp_ranges_for_prediction()
        candidates = [hp_ranges.from_ndarray(enc_config) for enc_config in inputs]
        resources = [self._fixed_resource] * len(candidates)
        prediction_list = []
        for _ in range(self._num_samples):
            self._model.resample()
            cost_values = self._model.sample_joint(candidates)
            prediction_list.append(
                np.array(
                    self._model.predict_times(
                        candidates=candidates,
                        resources=resources,
                        cost_values=cost_values,
                    )
                )
            )
        return [{"mean": np.mean(prediction_list, axis=0)}]

    def backward_gradient(
        self, input: np.ndarray, head_gradients: List[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """
        The gradient contribution through the cost model is blocked.

        """
        return [np.zeros_like(input)]

    def predict_mean_current_candidates(self) -> List[np.ndarray]:
        raise NotImplementedError()

    # We currently do not support cost models for the primary metric to be
    # optimized
    def current_best(self) -> List[np.ndarray]:
        raise NotImplementedError()


class CostEstimator(Estimator):
    """
    The name of the cost metric is ``model.cost_metric_name``.

    :param model: CostModel to be wrapped
    :param fixed_resource: :math:`c(x, r)` is predicted for this resource level r
    :param num_samples: Number of samples drawn in :meth:`predict`. Use this for
        random cost models only
    """

    def __init__(self, model: CostModel, fixed_resource: int, num_samples: int = 1):
        self._model = model
        self._num_samples = num_samples
        self._fixed_resource = None
        self.set_fixed_resource(fixed_resource)

    def get_params(self):
        return dict()

    def set_params(self, param_dict):
        pass

    @property
    def fixed_resource(self) -> int:
        return self._fixed_resource

    def set_fixed_resource(self, resource: int):
        assert resource >= 1, "Must be positive integer"
        self._fixed_resource = resource

    def fit_from_state(self, state: TuningJobState, update_params: bool) -> Predictor:
        """
        Models of type :class:`CostModel` do not have hyperparameters to be
        fit, so ``update_params`` is ignored here.

        """
        self._model.update(state)
        return CostFixedResourcePredictor(
            state=state,
            model=self._model,
            fixed_resource=self._fixed_resource,
            num_samples=self._num_samples,
        )
