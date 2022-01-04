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
import statsmodels.api as sm
import scipy.stats as sps

from syne_tune.optimizer.scheduler import TrialScheduler, \
    TrialSuggestion, SchedulerDecision
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.schedulers.searchers import BaseSearcher
import syne_tune.search_space as sp

__all__ = ['KernelDensityEstimator']

logger = logging.getLogger(__name__)


class KernelDensityEstimator(BaseSearcher):

    def __init__(
            self,
            configspace: Dict,
            metric: str,
            num_init_random_draws: int = 5,
            mode: str = "min",
            num_min_data_points: int = None,
            top_n_percent: int = 15,
            min_bandwidth: float = 1e-3,
            num_candidates: int = 64,
            bandwidth_factor: int = 3,
            points_to_evaluate: Optional[List[Dict]] = None,
            **kwargs
    ):
        super().__init__(configspace=configspace, metric=metric, points_to_evaluate=points_to_evaluate)
        assert num_init_random_draws >= 2
        self.mode = mode
        self.metric_name = metric
        self.num_evaluations = 0
        self.min_bandwidth = min_bandwidth
        self.num_minimum_observations = num_init_random_draws
        self.num_candidates = num_candidates
        self.bandwidth_factor = bandwidth_factor
        self.points_to_evaluate = points_to_evaluate
        self.num_min_data_points = len(configspace.keys()) if num_min_data_points is None else num_min_data_points
        assert self.num_min_data_points >= len(configspace.keys())
        self.top_n_percent = top_n_percent
        self.X = []
        self.y = []
        self.categorical_maps = {
            k: {cat: i for i, cat in enumerate(v.categories)}
            for k, v in configspace.items()
            if isinstance(v, sp.Categorical)
        }
        self.inv_categorical_maps = {
            hp: dict(zip(map.values(), map.keys())) for hp, map in self.categorical_maps.items()
        }

        self.good_kde = None
        self.bad_kde = None

        self.vartypes = []
        for hname in self.configspace:
            hp = self.configspace[hname]
            if isinstance(hp, sp.Categorical):
                self.vartypes.append(('u', len(hp.categories)))
            if isinstance(hp, sp.Integer):
                self.vartypes.append(('o', (hp.lower, hp.upper)))
            if isinstance(hp, sp.Float):
                self.vartypes.append(('c', 0))

    @staticmethod
    def to_feature(configspace, config, categorical_maps):
        def numerize(value, domain, categorical_map):
            if isinstance(domain, sp.Categorical):
                res = categorical_map[value] / len(domain)
                return res
            else:
                return [(value - domain.lower) / (domain.upper - domain.lower)]
        return np.hstack([
            numerize(value=config[k], domain=v, categorical_map=categorical_maps.get(k, {}))
            for k, v in configspace.items()
            if isinstance(v, sp.Domain)
        ])

    @staticmethod
    def from_feature(configspace, feature_vector, inv_categorical_maps):
        def inv_numerize(values, domain, categorical_map):
            if not isinstance(domain, sp.Domain):
                # constant value
                return domain
            else:
                if isinstance(domain, sp.Categorical):
                    index = int(values * len(domain))
                    return categorical_map[index]
                else:
                    if isinstance(domain, sp.Float):
                        return values * (domain.upper - domain.lower) + domain.lower
                    else:
                        return int(values * (domain.upper - domain.lower) + domain.lower)
        res = {}
        curr_pos = 0
        for k, domain in configspace.items():
            if hasattr(domain, "sample"):
                res[k] = domain.cast(
                    inv_numerize(
                        values=feature_vector[curr_pos],
                        domain=domain,
                        categorical_map=inv_categorical_maps.get(k, {})
                    )
                )
                curr_pos += 1
            else:
                res[k] = domain
        return res

    def to_objective(self, result: Dict):
        if self.mode == 'min':
            return result[self.metric_name]
        elif self.mode == 'max':
            return -result[self.metric_name]

    def _update(self, trial_id: str, config: Dict, result: Dict):
        self.X.append(self.to_feature(
            config=config,
            configspace=self.configspace,
            categorical_maps=self.categorical_maps,
        ))
        self.y.append(self.to_objective(result))
        self.train_kde(np.array(self.X), np.array(self.y))

    def get_config(self, **kwargs):
        # if not done with points_to_evaluate, pick the next one
        if self.points_to_evaluate is not None and self.num_evaluations < len(self.points_to_evaluate):
            suggestion = self.points_to_evaluate[self.num_evaluations]
        else:
            if self.good_kde is None and self.bad_kde is None:
                # if not enough suggestion made, sample randomly
                suggestion = {k: v.sample()
                    if isinstance(v, sp.Domain) else v for k, v in self.configspace.items()}
            else:
                l = self.good_kde.pdf
                g = self.bad_kde.pdf

                acquisition_function = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

                current_best = None
                val_current_best = None
                for i in range(self.num_candidates):
                    idx = np.random.randint(0, len(self.good_kde.data))
                    mean = self.good_kde.data[idx]
                    candidate = []

                    for m, bw, t in zip(mean, self.good_kde.bw, self.vartypes):
                        bw = max(bw, self.min_bandwidth)
                        vartype = t[0]
                        domain = t[1]
                        if vartype == 'c':
                            # continuous parameter
                            bw = self.bandwidth_factor * bw
                            candidate.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                        else:
                            # categorical or integer parameter
                            if np.random.rand() < (1 - bw):
                                candidate.append(m)
                            else:
                                if vartype == 'o':
                                    sample = np.random.randint(domain[0], domain[1])
                                    sample = (sample - domain[0]) / (domain[1] - domain[0])
                                    candidate.append(sample)
                                elif vartype == 'u':
                                    candidate.append(np.random.randint(domain) / domain)
                    val = acquisition_function(candidate)

                    if not np.isfinite(val):
                        logging.warning("candidate has non finite acquisition function value")

                    if val_current_best is None or val_current_best > val:
                        current_best = candidate
                        val_current_best = val

                suggestion = self.from_feature(
                    configspace=self.configspace,
                    feature_vector=current_best,
                    inv_categorical_maps=self.inv_categorical_maps)

        return suggestion

    def train_kde(self, train_data, train_targets):
        # split in good / poor data points
        n_good = max(self.num_min_data_points, (self.top_n_percent * train_data.shape[0]) // 100)
        n_bad = max(self.num_min_data_points, ((100 - self.top_n_percent) * train_data.shape[0]) // 100)

        idx = np.argsort(train_targets)

        train_data_good = train_data[idx[:n_good]]
        train_data_bad = train_data[idx[n_good:n_good + n_bad]]

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        types = [t[0] for t in self.vartypes]

        self.bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=types, bw='normal_reference')
        self.good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=types, bw='normal_reference')

        self.bad_kde.bw = np.clip(self.bad_kde.bw, self.min_bandwidth, None)
        self.good_kde.bw = np.clip(self.good_kde.bw, self.min_bandwidth, None)

    def clone_from_state(self, state):
        raise NotImplementedError
