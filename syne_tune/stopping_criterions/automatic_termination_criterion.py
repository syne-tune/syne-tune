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
import logging
import numpy as np
import torch
from typing import List, Dict, Any

from syne_tune.tuning_status import TuningStatus
from syne_tune.optimizer.schedulers.searchers.utils import (
    make_hyperparameter_ranges,
)
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

logger = logging.getLogger(__name__)

EPSILON_LOG = 1e-10
NOISE_LEVEL = 1e-3


class AutomaticTerminationCriterion(object):
    """
    Implements the automatic termination criterion proposed by Makarova et al. The idea is to automatically
    stop the optimization process if the expected gain in regret is smaller than the observation noise
    of the objective.

    Automatic Termination for Hyperparameter Optimization
    Anastasia Makarova and Huibin Shen and Valerio Perrone and Aaron Klein and Jean Baptiste Faddoul and Andreas Krause and Matthias Seeger and Cedric Archambeau
    First Conference on Automated Machine Learning (Main Track)
    2022

    :param config_space: Configuration space for evaluation function
    :param metric: The metric to be monitored.
    :param mode: The mode to select the top results ("min" or "max")
    :param threshold: The threshold on the regret. We stop the optimization process when it's unlikely that the expected decrease in regret compared to the global optimum is higher than this threshold.
    :param beta: Multiplier on the standard devication to compute the upper and lower confidence bound.
    :param seed: Seed for the random numer generator.
    :param warm_up: Number of iterations before we start fitting the GP. Especially, with little data, GP's are notoriously unreliable. Hence we only fit the GP once we collected some data points
    :param topq: We consider only the top q-% data points to fit the GP.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        threshold: float,
        mode: str = "min",
        beta: float = 1.0,
        seed: int = None,
        warm_up: int = 10,
        topq: float = 0.5,
    ):
        assert mode in ["min", "max"]

        self._mode = mode
        self._warm_up = warm_up
        self._seed = np.random.randint(2**32 - 1) if seed is None else seed
        self._metric = metric
        self._topq = topq
        self._beta = beta
        self._threshold = threshold
        self._config_space = config_space
        self._hp_ranges = make_hyperparameter_ranges(config_space)

        if self._mode == "max":
            self.multiplier = 1
        else:
            self.multiplier = -1

    def _config_to_feature_matrix(self, configs: List[dict]) -> Tensor:
        bounds = Tensor(self._hp_ranges.get_ndarray_bounds()).T
        X = Tensor(self._hp_ranges.to_ndarray_matrix(configs))
        return normalize(X, bounds)

    def __call__(self, status: TuningStatus) -> bool:
        """Return a boolean representing if the tuning has to stop."""

        trials = status.trial_rows

        if len(trials) == 0:
            return False

        evaluations = []
        observations = []
        for trial in trials.values():
            if self._metric not in trial:
                continue
            y = self.multiplier * trial[self._metric]
            evaluations.append(y)

            observation = {}
            for hp in self._config_space:
                observation[hp] = trial[hp]
            observations.append(observation)

        if len(observations) < self._warm_up:
            return False

        observations = self._config_to_feature_matrix(observations).double()
        evaluations = torch.tensor(evaluations).double()

        noise_std = NOISE_LEVEL
        evaluations += noise_std * torch.randn_like(evaluations)

        n = len(observations)

        if self._topq is not None:
            topn = max(20, int(n * self._topq))
            top_inds = torch.argsort(evaluations, descending=True, dim=0)[:topn]
            evaluations = evaluations[top_inds]
            observations = observations[top_inds]

        if len(observations.shape) == 1:
            observations = observations.reshape((-1, 1))
        if len(evaluations.shape) == 1:
            evaluations = evaluations.reshape((-1, 1))

        gp = SingleTaskGP(observations, evaluations)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll, max_attempts=1)
        acq = UpperConfidenceBound(
            model=gp,
            beta=self._beta,
        )
        _, max_ucb = optimize_acqf(
            acq,
            bounds=Tensor(self._hp_ranges.get_ndarray_bounds()).T,
            q=1,
            num_restarts=3,
            raw_samples=100,
        )
        posterior = gp.posterior(observations)
        mean = posterior.mean
        std = torch.sqrt(posterior.variance)
        lcb = mean - self._beta * std
        max_lcb = lcb.max()

        upper_bound = max_ucb - max_lcb

        stop = upper_bound < self._threshold

        return stop


if __name__ == "__main__":

    from syne_tune.backend.trial_status import Trial, Status
    from syne_tune.config_space import uniform

    metric = "loss"
    config_space = {"x": uniform(0, 1)}

    stop_criterion = AutomaticTerminationCriterion(
        config_space, threshold=0.9, metric=metric, mode="min"
    )

    status = TuningStatus(metric_names=[metric])

    trial_status_dict = {}
    new_results = []
    for i in range(20):
        x = np.random.rand()
        trial = Trial(trial_id=i, config={"x": x}, creation_time=None)
        trial_status_dict[i] = (trial, Status.completed)
        new_results.append((i, {metric: (x - 0.5) ** 2}))
    status.update(trial_status_dict, new_results)
    # status.update(
    #     trial_status_dict={
    #         0: (trial0, Status.completed),
    #         1: (trial1, Status.completed),
    #     },
    #     new_results=[
    #         (0, {metric: 4.0}),
    #         (1, {metric: 3.0}),
    #     ],
    # )
    assert not stop_criterion(status)
