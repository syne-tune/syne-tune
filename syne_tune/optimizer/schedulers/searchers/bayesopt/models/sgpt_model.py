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
from typing import List, Optional, Dict

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import FantasizedPendingEvaluation, \
    ConfigurationFilter, TrialEvaluations
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import GPModel, GaussProcSurrogateModel, \
    GaussProcEmpiricalBayesModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import SurrogateModel
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningMixin, TransferLearningTaskEvaluations
import numpy as np


class ScalableGaussianProcessTransferModelFactory(TransferLearningMixin, GaussProcEmpiricalBayesModelFactory):
    def __init__(self, config_space: Dict, transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
                 metric: str, source_gp_models: List[GPModel], gpmodel: GPModel, **kwargs):
        super().__init__(config_space=config_space, transfer_learning_evaluations=transfer_learning_evaluations,
                         metric_names=[metric], gpmodel=gpmodel, **kwargs)
        self._source_surrogates = list()
        for i, task_data in enumerate(transfer_learning_evaluations.values()):
            scores = task_data.objective_values(metric).mean(axis=1)[:, -1].tolist()
            hp_ranges = make_hyperparameter_ranges(config_space)
            config_for_trial = {str(key): value for key, value in task_data.hyperparameters.to_dict('index').items()}
            trials_evaluations = [TrialEvaluations(str(trial_id), metrics={self.active_metric: score})
                                  for trial_id, score in enumerate(scores)]
            state = TuningJobState(hp_ranges=hp_ranges, config_for_trial=config_for_trial,
                                   trials_evaluations=trials_evaluations)
            factory = GaussProcEmpiricalBayesModelFactory(gpmodel=source_gp_models[i],
                                                          num_fantasy_samples=kwargs.get('num_fantasy_samples'),
                                                          active_metric=self.active_metric,
                                                          normalize_targets=kwargs.get('normalize_targets'),
                                                          no_fantasizing=True)
            factory._posterior_for_state(state, False)
            surrogate = factory.model(state, True)
            self._source_surrogates.append(surrogate)

    def model(self, state: TuningJobState, fit_params: bool) -> SurrogateModel:
        return ScalableGaussianProcessTransfer(source_surrogates=self._source_surrogates,
                                               **self._model_kwargs(state=state, fit_params=fit_params))


class ScalableGaussianProcessTransfer(GaussProcSurrogateModel):
    """
    TODO:
    """

    def __init__(
            self, state: TuningJobState,
            gpmodel: GPModel,
            source_surrogates: List[GaussProcSurrogateModel],
            fantasy_samples: List[FantasizedPendingEvaluation],
            active_metric: str = None,
            normalize_mean: float = 0.0, normalize_std: float = 1.0,
            filter_observed_data: Optional[ConfigurationFilter] = None):
        super().__init__(state=state, gpmodel=gpmodel, fantasy_samples=fantasy_samples,
                         active_metric=active_metric, normalize_mean=normalize_mean,
                         normalize_std=normalize_std,
                         filter_observed_data=filter_observed_data)
        self._source_surrogates = source_surrogates
        self._weights = [1.0 / len(self._source_surrogates) for _ in range(len(self._source_surrogates))]

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        target_predictions = super().predict(inputs=inputs)
        if len(self._source_surrogates):
            source_pred = [surrogate.predict(inputs=inputs) for surrogate in self._source_surrogates]
            weighted_source_pred = [np.sum([w * sp[i]['mean'] for sp, w in zip(source_pred, self._weights)], axis=0)
                                    for i in range(len(source_pred[0]))]
            for tp, sp in zip(target_predictions, weighted_source_pred):
                assert tp['mean'].shape == sp.shape
                tp['mean'] += sp
        return target_predictions

    def backward_gradient(
            self, input: np.ndarray,
            head_gradients: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        target_grad = super().backward_gradient(input=input, head_gradients=head_gradients)
        if len(self._source_surrogates):
            source_head_gradients = [{'mean': hg['mean']} for hg in head_gradients]
            source_grad = [surrogate.backward_gradient(input=input, head_gradients=source_head_gradients)
                           for surrogate in self._source_surrogates]
            weighted_source_grad = [np.sum([w * sg[i] for sg, w in zip(source_grad, self._weights)], axis=0)
                                    for i in range(len(source_grad[0]))]
            target_grad = [tg + sg for tg, sg in zip(target_grad, weighted_source_grad)]
        return target_grad
