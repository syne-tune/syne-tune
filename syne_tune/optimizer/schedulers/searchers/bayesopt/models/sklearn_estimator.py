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
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator import (
    Estimator,
    transform_state_to_data,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.estimator import (
    SklearnEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.sklearn_predictor import (
    SklearnPredictorWrapper,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    Predictor,
)


class SklearnEstimatorWrapper(Estimator):
    """
    Wrapper class for the sklearn estimators to be used with BayesianOptimizationSearcher
    """

    def __init__(self, contributed_estimator: SklearnEstimator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contributed_estimator = contributed_estimator

    @property
    def normalize_targets(self) -> bool:
        """
        :return: Should targets in ``state`` be normalized before fitting the estimator
        """
        return self.contributed_estimator.normalize_targets

    def fit_from_state(
        self, state: TuningJobState, update_params: bool = False
    ) -> Predictor:
        """
        Creates a
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes.Predictor`
        object based on data in ``state``.

        If the model also has hyperparameters, these are learned iff
        ``update_params == True``. Otherwise, these parameters are not changed,
        but only the posterior state is computed.
        If your surrogate model is not Bayesian, or does not have hyperparameters,
        you can ignore the ``update_params`` argument.

        If ``self.state.pending_evaluations`` is not empty, we compute posterior for state without pending evals.
        This method can be overriten for any other behaviour such as one found in
        ``syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model.GaussProcEstimator.fit_from_state``

        :param state: Current data model parameters are to be fit on, and the
            posterior state is to be computed from
        :param update_params: See above
        :return: Predictor, wrapping the posterior state
        """
        if state.pending_evaluations:
            state = TuningJobState(
                hp_ranges=state.hp_ranges,
                config_for_trial=state.config_for_trial,
                trials_evaluations=state.trials_evaluations,
                failed_trials=state.failed_trials,
            )

        data = transform_state_to_data(
            state=state, normalize_targets=self.normalize_targets
        )
        contributed_predictor = self.contributed_estimator.fit(
            data.features, data.targets, update_params=update_params
        )
        return SklearnPredictorWrapper(
            contributed_predictor=contributed_predictor, state=state
        )
