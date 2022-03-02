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
from typing import Dict, Optional, List
import logging
import numpy as np

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.utils.errors import NotPSDError

from syne_tune.optimizer.scheduler import TrialScheduler, \
    TrialSuggestion, SchedulerDecision
from syne_tune.backend.trial_status import Trial
import syne_tune.config_space as cs

__all__ = ['BotorchGP']

logger = logging.getLogger(__name__)


class BotorchGP(TrialScheduler):

    def __init__(
            self,
            config_space: Dict,
            metric: str,
            num_init_random_draws: int = 5,
            mode: str = "min",
            points_to_evaluate: Optional[List[Dict]] = None
    ):
        """
        :param config_space:
        :param metric: metric to optimize.
        :param num_init_random_draws: number of initial random draw, after this number the suggestion are obtained
        using the posterior of a GP built on available observations.
        :param mode: 'min' or 'max'
        :param points_to_evaluate: points to evaluate first
        """
        super().__init__(config_space)
        assert num_init_random_draws >= 2
        assert mode in ['min', 'max']
        self.mode = mode
        self.metric_name = metric
        self.num_evaluations = 0
        self.num_minimum_observations = num_init_random_draws
        self.points_to_evaluate = points_to_evaluate
        self.X = []
        self.y = []
        self.categorical_maps = {
            k: {cat: i for i, cat in enumerate(v.categories)}
            for k, v in config_space.items()
            if isinstance(v, cs.Categorical)
        }
        self.inv_categorical_maps = {
            hp: dict(zip(map.values(), map.keys())) for hp, map in self.categorical_maps.items()
        }

    def on_trial_complete(self, trial: Trial, result: Dict):
        # update available observations with final result.
        self.X.append(self._encode_config(
            config=trial.config,
            config_space=self.config_space,
            categorical_maps=self.categorical_maps,
        ))
        self.y.append(result[self.metric_name])

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        if self.points_to_evaluate is not None and self.num_evaluations < len(self.points_to_evaluate):
            # if we are not done yet with points_to_evaluate, we pick the next one from this list
            suggestion = self.points_to_evaluate[self.num_evaluations]
        else:
            enough_suggestion = len(self.y) < self.num_minimum_observations
            if enough_suggestion:
                # if not enough suggestion made, sample randomly
                suggestion = self.sample_random()
            else:
                suggestion = self.sample_gp()

        self.num_evaluations += 1
        return TrialSuggestion.start_suggestion(config=suggestion)

    def sample_random(self) -> Dict:
        return {
            k: v.sample()
            if isinstance(v, cs.Domain) else v
            for k, v in self.config_space.items()
        }

    def sample_gp(self) -> Dict:
        try:
            # First updates GP and compute its posterior, then maximum acquisition function to find candidate.
            # todo normalize input data
            train_X = torch.Tensor(self.X)
            train_Y = standardize(torch.Tensor(self.y).reshape(-1, 1))

            self.gp = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_model(mll)

            # ask candidate to GP by maximizing its acquisition function.
            UCB = UpperConfidenceBound(
                self.gp,
                beta=0.1,
                maximize=self.mode == 'max'
            )

            bounds = torch.stack([train_X.min(axis=0).values, train_X.max(axis=0).values])
            candidate, acq_value = optimize_acqf(
                UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )

            return BotorchGP._decode_config(
                config_space=self.config_space,
                encoded_vector=candidate.detach().numpy()[0],
                inv_categorical_maps=self.inv_categorical_maps,
            )

        except NotPSDError as e:
            # In case Cholesky inversion fails, we sample randomly.
            logger.info("Chlolesky failed, sampling randomly.")
            return self.sample_random()

    @staticmethod
    def _encode_config(config_space: Dict, config: Dict, categorical_maps: Dict) -> np.array:
        """
        Encode a configuration into a vector that can be decoded back with `_decode_config`.
        :param config_space:
        :param config: configuration to be encoded
        :param categorical_maps: dictionary from categorical Hyperparameter name to a dictionary mapping categories to
        an integer index. For instance {"cell_type": {"conv3x3": 0, "skip": 1}}
        :return: encoded vector.
        """
        def numerize(value, domain, categorical_map):
            if isinstance(domain, cs.Categorical):
                res = np.zeros(len(domain))
                res[categorical_map[value]] = 1
                return res
            else:
                if hasattr(domain, "lower") and hasattr(domain, "upper"):
                    return [(value - domain.lower) / (domain.upper - domain.lower)]
                else:
                    return [value]
        return np.hstack([
            numerize(value=config[k], domain=v, categorical_map=categorical_maps.get(k, {}))
            for k, v in config_space.items()
            if isinstance(v, cs.Domain)
        ])

    @staticmethod
    def _decode_config(config_space: Dict, encoded_vector: np.array, inv_categorical_maps: Dict) -> Dict:
        """
        Return a config dictionary given an encoded vector.
        :param config_space:
        :param encoded_vector:
        :param inv_categorical_maps: dictionary from each categorical Hyperparameter name to a dictionary maping
        category index to category value. For instance {"cell_type": {0: "conv3x3", 1: "skip"}}
        :return:
        """
        def inv_numerize(values, domain, categorical_map):
            if not isinstance(domain, cs.Domain):
                # constant value
                return domain
            else:
                if isinstance(domain, cs.Categorical):
                    values = 1.0 * (values == values.max())
                    index = max(np.arange(len(domain)) * values)
                    return categorical_map[index]
                else:
                    if hasattr(domain, "lower") and hasattr(domain, "upper"):
                        return values[0] * (domain.upper - domain.lower) + domain.lower
                    else:
                        return values[0]
        cur_pos = 0
        res = {}
        for k, domain in config_space.items():
            if hasattr(domain, "sample"):
                length = len(domain) if isinstance(domain, cs.Categorical) else 1
                res[k] = domain.cast(
                    inv_numerize(
                        values=encoded_vector[cur_pos:cur_pos + length],
                        domain=domain,
                        categorical_map=inv_categorical_maps.get(k, {})
                    )
                )
                cur_pos += length
            else:
                res[k] = domain
        return res

    def metric_names(self) -> List[str]:
        return [self.metric_name]

    def metric_mode(self) -> str:
        return self.mode