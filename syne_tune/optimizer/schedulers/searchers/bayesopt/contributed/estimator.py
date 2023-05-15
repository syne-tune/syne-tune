import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.contributed.predictor import ContributedPredictor, \
    ContributedPredictorWrapper
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator import Estimator, transform_state_to_data
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import Predictor


class ContributedEstimator:
    """
    Base class for the contributed Estimators
    """

    def fit(self, X: np.ndarray, y: np.ndarray, update_params: bool) -> ContributedPredictor:
        """
        Implements :meth:`fit_from_state`, given transformed data.

        :param X: Training data in ndarray of shape (n_samples, n_features)
        :param y: Target values in ndarray of shape (n_samples,)
        :param update_params: Should model (hyper)parameters be updated?
        :return: Predictor, wrapping the posterior state
        """
        raise NotImplementedError()

    @property
    def normalize_targets(self) -> bool:
        """
        :return: Should targets in ``state`` be normalized before calling
            :meth:`fit`?
        """
        return False


class ContributedEstimatorWrapper(Estimator):
    """
    Wrapper class for the contributed estimators to be used with BayesianOptimizationSearcher
    """

    def __init__(
            self,
            contributed_estimator: ContributedEstimator,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.contributed_estimator = contributed_estimator

    @property
    def normalize_targets(self) -> bool:
        """
        :return: Should targets in ``state`` be normalized before fitting the estimator
        """
        return self.contributed_estimator.normalize_targets

    def fit_from_state(self, state: TuningJobState, update_params: bool = False) -> Predictor:
        """
        Creates a
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`
        object based on data in ``state``.

        If the model also has hyperparameters, these are learned iff
        ``update_params == True``. Otherwise, these parameters are not changed,
        but only the posterior state is computed.

        If your surrogate model is not Bayesian, or does not have hyperparameters,
        you can ignore the ``update_params`` argument,

        :param state: Current data model parameters are to be fit on, and the
            posterior state is to be computed from
        :param update_params: See above
        :return: Predictor, wrapping the posterior state
        """
        data = transform_state_to_data(
            state=state, normalize_targets=self.normalize_targets
        )
        contributed_predictor = self.contributed_estimator.fit(
            data.features,
            data.targets,
            update_params=update_params
        )
        return ContributedPredictorWrapper(
            contributed_predictor=contributed_predictor,
            state=state

        )
