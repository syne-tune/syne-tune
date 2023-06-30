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
from typing import Dict, Optional, Iterable, List, Callable

import numpy as np
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    Predictor,
    ScoringFunction,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration


class MultiObjectiveLCBRandomLinearScalarization(ScoringFunction):
    """
    Note: This is the multi objective random scalarization scoring function based on the work of Biswajit et al. [1].
    This scoring function uses Lower Confidence Bound as the acquisition for the scalarized objective
    :math:`h(\mu, \sigma) = \mu - \kappa * \sigma`

        | [1] Paria, Biswajit, Kirthevasan Kandasamy and Barnabás Póczos.
        | A Flexible Framework for Multi-Objective Bayesian Optimization using Random Scalarizations.
        | Conference on Uncertainty in Artificial Intelligence (2018).

    :param predictor: Surrogate predictor for statistics of predictive distribution
    :param weights_sampler: Callable that can generate weights for each objective.
        Once called it will return a dictionary mapping metric name to scalarization weight as
        {
            <name of metric 1> : <weight for metric 1>,
            <name of metric 2> : <weight for metric 2>,
            ...
        }
    :param kappa: Hyperparameter used for the LCM portion of the scoring
    :param normalize_acquisition: If True, use rank-normalization on the acquisition function results before weighting.
    :param random_seed: The random seed used for default weights_sampler if not provided.
    """

    def __init__(
        self,
        predictor: Dict[str, Predictor],
        active_metric: Optional[List[str]] = None,
        weights_sampler: Optional[Callable[[], Dict[str, float]]] = None,
        kappa: float = 0.5,
        normalize_acquisition: bool = True,
        random_seed: int = None,
    ):
        super(MultiObjectiveLCBRandomLinearScalarization, self).__init__(
            predictor, active_metric
        )
        self.kappa = kappa
        self.normalize_acquisition = normalize_acquisition

        if weights_sampler is None:
            state = RandomState(random_seed)

            def weights_sampler():
                return {name: state.uniform() for name in predictor.keys()}

        self.weights_sampler = weights_sampler

    def score(
        self,
        candidates: Iterable[Configuration],
        predictor: Optional[Dict[str, Predictor]] = None,
    ) -> List[float]:
        from scipy import stats

        if predictor is None:
            predictor = self.predictor
        weights = self.weights_sampler()
        scores = np.zeros(len(candidates))
        for metric, metric_predictor in predictor.items():
            predictions = metric_predictor.predict_candidates(candidates)
            predicted_mean = predictions[0]["mean"]
            predicted_std = predictions[0]["std"]

            metric_score = self._single_objective_score(predicted_mean, predicted_std)
            if self.normalize_acquisition:
                metrics_score_normalized = stats.rankdata(metric_score)
                metrics_score_normalized = (
                    metrics_score_normalized / metrics_score_normalized.max()
                )
            scores += metrics_score_normalized * weights[metric]
        return list(scores)

    def _single_objective_score(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Acquisition function for a single objective
        """
        return mean - std * self.kappa
