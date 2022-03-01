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

import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import FantasizedPendingEvaluation, \
    ConfigurationFilter, TrialEvaluations
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import GPModel, GaussProcSurrogateModel, \
    GaussProcEmpiricalBayesModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import SurrogateModel
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningMixin, TransferLearningTaskEvaluations


class ScalableGaussianProcessTransferModelFactory(TransferLearningMixin, GaussProcEmpiricalBayesModelFactory):
    def __init__(self, config_space: Dict, transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
                 metric: str, source_gp_models: List[GPModel], gpmodel: GPModel, bandwidth: float, sample_size: int,
                 **kwargs):
        super().__init__(config_space=config_space, transfer_learning_evaluations=transfer_learning_evaluations,
                         metric_names=[metric], gpmodel=gpmodel, **kwargs)
        self._source_surrogates = list()
        self._bandwidth = bandwidth
        for i, task_data in enumerate(transfer_learning_evaluations.values()):
            idx = np.arange(len(task_data.hyperparameters))
            idx = np.random.choice(idx, size=sample_size, replace=False)
            scores = task_data.objective_values(metric).mean(axis=1)[idx, -1].tolist()
            hp_ranges = make_hyperparameter_ranges(config_space)
            config_for_trial = {str(key): value for key, value in enumerate(task_data.hyperparameters.iloc[idx]
                                                                            .to_dict('index').values())}
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
        self._source_meta_features = [list() for _ in range(len(self._source_surrogates))]
        self._target_meta_features = list()

    def model(self, state: TuningJobState, fit_params: bool) -> SurrogateModel:
        target_surrogate = ScalableGaussianProcessTransfer(
            source_surrogates=self._source_surrogates, bandwidth=self._bandwidth,
            **self._model_kwargs(state=state, fit_params=fit_params))
        if len(self._source_surrogates):
            candidates, _ = state.observed_data_for_metric(metric_name=self.active_metric)
            features = state.hp_ranges.to_ndarray_matrix(candidates)[-1].reshape(1, -1)
            for surrogate, meta_feature in zip(self._source_surrogates, self._source_meta_features):
                predictions = surrogate.predict(features)
                meta_feature.append(np.mean([p['mean'] for p in predictions]))
            self._target_meta_features.append(state.trials_evaluations[-1].metrics[self.active_metric])
            target_surrogate.estimate_weights(self._source_meta_features, self._target_meta_features)
        return target_surrogate


class ScalableGaussianProcessTransfer(GaussProcSurrogateModel):

    def __init__(
            self,
            state: TuningJobState,
            gpmodel: GPModel,
            source_surrogates: List[GaussProcSurrogateModel],
            fantasy_samples: List[FantasizedPendingEvaluation],
            bandwidth: float,
            active_metric: str = None,
            normalize_mean: float = 0.0,
            normalize_std: float = 1.0,
            filter_observed_data: Optional[ConfigurationFilter] = None):
        """
        A transfer learning method which uses a product of experts. One Gaussian process is learned for each task
        (source and target tasks) and its predicted mean is simply the weighted sum of the experts' predictions. The
        predicted standard deviation is the predicted standard deviation of the expert for the target task only.
        The weights are estimated based on meta-features describing the tasks. Here, we implement the ranking-based
        meta-features, i.e. the ranking agreement of hyperparameter configurations across tasks.

        Reference: Scalable Gaussian Process based Transfer Surrogates for Hyperparameter Optimization.
        Martin Wistuba, Nicolas Schilling, Lars Schmidt-Thieme. Machine Learning volume 107, pages 43–78, 2018.

        :param state: Current optimization state.
        :param gpmodel: GP model used as an expert for the target task.
        :param source_surrogates: List of experts for the source tasks.
        :param fantasy_samples: See `GaussProcSurrogateModel`.
        :param bandwidth: A hyperparameter of this method which is part of the kernel function which estimates task
        similarity. Values somewhere in the range of 0.1 ad 0.5 are reasonable.
        :param active_metric: See `GaussProcSurrogateModel`.
        :param normalize_mean: See `GaussProcSurrogateModel`.
        :param normalize_std: See `GaussProcSurrogateModel`.
        :param filter_observed_data: See `GaussProcSurrogateModel`.
        """
        super().__init__(state=state, gpmodel=gpmodel, fantasy_samples=fantasy_samples,
                         active_metric=active_metric, normalize_mean=normalize_mean,
                         normalize_std=normalize_std,
                         filter_observed_data=filter_observed_data)
        self._source_surrogates = source_surrogates
        self._bandwidth = bandwidth
        self._weights = None
        self._target_weight = None

    def estimate_weights(self, source_meta_features: List[List[float]], target_meta_features: List[float]):
        self._weights = list()
        self._target_weight = 0.75
        denominator = self._target_weight
        for meta_features in source_meta_features:
            discordant_pairs = 0
            total_pairs = 0
            for i in range(len(target_meta_features)):
                for j in range(len(target_meta_features)):
                    if (target_meta_features[i] < target_meta_features[j]) ^ (meta_features[i] < meta_features[j]):
                        discordant_pairs += 1
                    total_pairs += 1
            t = discordant_pairs / total_pairs / self._bandwidth
            t = 0.75 * (1 - t * t) if t < 1 else 0
            self._weights.append(t)
            denominator += t
        self._weights = np.array(self._weights) / denominator
        self._target_weight /= denominator

    def predict(self, inputs: np.ndarray) -> List[Dict[str, np.ndarray]]:
        target_predictions = super().predict(inputs=inputs)
        if len(self._source_surrogates):
            for tp in target_predictions:
                tp['mean'] *= self._target_weight
            source_pred = [surrogate.predict(inputs=inputs) for surrogate in self._source_surrogates]
            weighted_source_pred = [np.sum([w * sp[i]['mean'] for sp, w in zip(source_pred, self._weights)], axis=0)
                                    for i in range(len(source_pred[0]))]
            for tp, sp in zip(target_predictions, weighted_source_pred):
                tp['mean'] += sp if len(tp['mean'].shape) == 1 else sp[:, None]
        return target_predictions

    def backward_gradient(
            self, input: np.ndarray,
            head_gradients: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        target_grad = super().backward_gradient(input=input, head_gradients=head_gradients)
        if len(self._source_surrogates):
            source_head_gradients = [{'mean': hg['mean'] if len(hg['mean'].shape) == 1 else np.mean(hg['mean'], axis=1)}
                                     for hg in head_gradients]
            source_grad = [surrogate.backward_gradient(input=input, head_gradients=source_head_gradients)
                           for surrogate in self._source_surrogates]
            weighted_source_grad = [np.sum([w * sg[i] for sg, w in zip(source_grad, self._weights)], axis=0)
                                    for i in range(len(source_grad[0]))]
            target_grad = [self._target_weight * tg + sg for tg, sg in zip(target_grad, weighted_source_grad)]
        return target_grad
