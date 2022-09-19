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
from typing import Optional, List
import logging
import numpy as np
import statsmodels.api as sm
import scipy.stats as sps

from syne_tune.optimizer.schedulers.searchers import SearcherWithRandomSeed
import syne_tune.config_space as sp
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)

__all__ = ["KernelDensityEstimator"]

logger = logging.getLogger(__name__)


class KernelDensityEstimator(SearcherWithRandomSeed):
    """
    Fits two kernel density estimators (KDE) to model the density of the top N
    configurations as well as the density of the configurations that are not
    among the top N, respectively. New configurations are sampled by optimizing
    the ratio of these two densities. KDE as model for Bayesian optimization has
    been originally proposed by Bergstra et al. Compared to their original
    implementation TPE, we use multi-variate instead of univariate KDE, as
    proposed by Falkner et al.
    Code is based on the implementation by Falkner et al:
    https://github.com/automl/HpBandSter/tree/master/hpbandster

    Algorithms for Hyper-Parameter Optimization
    J. Bergstra and R. Bardenet and Y. Bengio and B. K{\'e}gl
    Proceedings of the 24th International Conference on Advances in Neural
    Information Processing Systems

    BOHB: Robust and Efficient Hyperparameter Optimization at Scale
    S. Falkner and A. Klein and F. Hutter
    Proceedings of the 35th International Conference on Machine Learning

    Parameters
    ----------
    config_space: dict
        Configuration space for trial evaluation function
    metric : str
        Name of metric to optimize, key in result's obtained via
        `on_trial_result`
    mode : str
        Mode to use for the metric given, can be 'min' or 'max'. Is obtained
        scheduler in `configure_scheduler`
    random_seed_generator : RandomSeedGenerator (optional)
        If given, the random_seed for `random_state` is obtained from there,
        otherwise `random_seed` is used
    random_seed : int (optional)
        This is used if `random_seed_generator` is not given.
    num_min_data_points: int
        Minimum number of data points that we use to fit the KDEs. If set to None
        than we set this to the number of hyperparameters.
    top_n_percent: int
        Determines how many datapoints we use to fit the first KDE model for
        modeling the well performing configurations.
    min_bandwidth: float
        The minimum bandwidth for the KDE models
    num_candidates: int
        Number of candidates that are sampled to optimize the acquisition function
    bandwidth_factor: int
        We sample continuous hyperparameter from a truncated Normal. This factor
        is multiplied to the bandwidth to define the standard deviation of this
        trunacted Normal.
    random_fraction: float
        Defines the fraction of configurations that are drawn uniformly at random
        instead of sampling from the model
    points_to_evaluate: List[dict] or None
        List of configurations to be evaluated initially (in that order).
        Each config in the list can be partially specified, or even be an
        empty dict. For each hyperparameter not specified, the default value
        is determined using a midpoint heuristic.
        If None (default), this is mapped to [dict()], a single default config
        determined by the midpoint heuristic. If [] (empty list), no initial
        configurations are specified.
    """

    def __init__(
        self,
        config_space: dict,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        mode: str = "min",
        num_min_data_points: int = None,
        top_n_percent: int = 15,
        min_bandwidth: float = 1e-3,
        num_candidates: int = 64,
        bandwidth_factor: int = 3,
        random_fraction: float = 0.33,
        **kwargs,
    ):
        super().__init__(
            config_space=config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            **kwargs,
        )
        self.mode = mode
        self.num_evaluations = 0
        self.min_bandwidth = min_bandwidth
        self.random_fraction = random_fraction
        self.num_candidates = num_candidates
        self.bandwidth_factor = bandwidth_factor
        self.top_n_percent = top_n_percent
        self.X = []
        self.y = []
        self.categorical_maps = {
            k: {cat: i for i, cat in enumerate(v.categories)}
            for k, v in config_space.items()
            if isinstance(v, sp.Categorical)
        }
        self.inv_categorical_maps = {
            hp: dict(zip(map.values(), map.keys()))
            for hp, map in self.categorical_maps.items()
        }

        self.good_kde = None
        self.bad_kde = None

        self.vartypes = []

        for name, hp in self.config_space.items():
            if isinstance(hp, sp.Categorical):
                self.vartypes.append(("u", len(hp.categories)))
            elif isinstance(hp, sp.Integer):
                self.vartypes.append(("o", (hp.lower, hp.upper)))
            elif isinstance(hp, sp.Float):
                self.vartypes.append(("c", 0))
            elif isinstance(hp, sp.FiniteRange):
                if hp.cast_int:
                    self.vartypes.append(("o", (hp.lower, hp.upper)))
                else:
                    self.vartypes.append(("c", 0))

        self.num_min_data_points = (
            len(self.vartypes) if num_min_data_points is None else num_min_data_points
        )
        assert self.num_min_data_points >= len(self.vartypes)
        self._resource_attr = kwargs.get("resource_attr")
        # Debug log printing (switched on by default)
        debug_log = kwargs.get("debug_log", True)
        if isinstance(debug_log, bool):
            if debug_log:
                self._debug_log = DebugLogPrinter()
            else:
                self._debug_log = None
        else:
            assert isinstance(debug_log, DebugLogPrinter)
            self._debug_log = debug_log

    def to_feature(self, config):
        def numerize(value, domain, categorical_map):
            if isinstance(domain, sp.Categorical):
                res = categorical_map[value] / len(domain)
                return res
            elif isinstance(domain, sp.Float):
                return [(value - domain.lower) / (domain.upper - domain.lower)]
            elif isinstance(domain, sp.FiniteRange):
                if domain.cast_int:
                    a = 1 / (2 * (domain.upper - domain.lower + 1))
                    b = domain.upper
                    return [(value - a) / (b - a)]
                else:
                    return [(value - domain.lower) / (domain.upper - domain.lower)]
            elif isinstance(domain, sp.Integer):
                a = 1 / (2 * (domain.upper - domain.lower + 1))
                b = domain.upper
                return [(value - a) / (b - a)]

        return np.hstack(
            [
                numerize(
                    value=config[k],
                    domain=v,
                    categorical_map=self.categorical_maps.get(k, {}),
                )
                for k, v in self.config_space.items()
                if isinstance(v, sp.Domain)
            ]
        )

    def from_feature(self, feature_vector):
        def inv_numerize(values, domain, categorical_map):
            if not isinstance(domain, sp.Domain):
                # constant value
                return domain
            else:
                if isinstance(domain, sp.Categorical):
                    index = int(values * len(domain))
                    return categorical_map[index]
                elif isinstance(domain, sp.Float):
                    return values * (domain.upper - domain.lower) + domain.lower
                elif isinstance(domain, sp.FiniteRange):
                    if domain.cast_int:
                        a = 1 / (2 * (domain.upper - domain.lower + 1))
                        b = domain.upper
                        return np.ceil(values * (b - a) + a)
                    else:
                        return values * (domain.upper - domain.lower) + domain.lower
                elif isinstance(domain, sp.Integer):
                    a = 1 / (2 * (domain.upper - domain.lower + 1))
                    b = domain.upper
                    return np.ceil(values * (b - a) + a)

        res = {}
        curr_pos = 0
        for k, domain in self.config_space.items():
            if isinstance(domain, sp.Domain):
                res[k] = domain.cast(
                    inv_numerize(
                        values=feature_vector[curr_pos],
                        domain=domain,
                        categorical_map=self.inv_categorical_maps.get(k, {}),
                    )
                )
                curr_pos += 1
            else:
                res[k] = domain
        return res

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

        assert isinstance(
            scheduler, FIFOScheduler
        ), "This searcher requires FIFOScheduler scheduler"
        super().configure_scheduler(scheduler)
        self.mode = scheduler.mode

    def to_objective(self, result: dict):
        if self.mode == "min":
            return result[self._metric]
        elif self.mode == "max":
            return -result[self._metric]

    def _update(self, trial_id: str, config: dict, result: dict):
        self.X.append(self.to_feature(config=config))
        self.y.append(self.to_objective(result))
        if self._debug_log is not None:
            metric_val = result[self._metric]
            if self._resource_attr is not None:
                # For HyperbandScheduler, also add the resource attribute
                resource = int(result[self._resource_attr])
                trial_id = trial_id + ":{}".format(resource)
            msg = f"Update for trial_id {trial_id}: metric = {metric_val:.3f}"
            logger.info(msg)

    def get_config(self, **kwargs):
        suggestion = self._next_initial_config()

        if suggestion is None:
            models = self.train_kde(np.array(self.X), np.array(self.y))

            if models is None or self.random_state.rand() < self.random_fraction:
                # return random candidate because a) we don't have enough data points or
                # b) we sample some fraction of all samples randomly
                suggestion = {
                    k: v.sample() if isinstance(v, sp.Domain) else v
                    for k, v in self.config_space.items()
                }
            else:
                self.bad_kde = models[0]
                self.good_kde = models[1]
                l = self.good_kde.pdf
                g = self.bad_kde.pdf

                acquisition_function = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

                current_best = None
                val_current_best = None
                for i in range(self.num_candidates):
                    idx = self.random_state.randint(0, len(self.good_kde.data))
                    mean = self.good_kde.data[idx]
                    candidate = []

                    for m, bw, t in zip(mean, self.good_kde.bw, self.vartypes):
                        bw = max(bw, self.min_bandwidth)
                        vartype = t[0]
                        domain = t[1]
                        if vartype == "c":
                            # continuous parameter
                            bw = self.bandwidth_factor * bw
                            candidate.append(
                                sps.truncnorm.rvs(
                                    -m / bw,
                                    (1 - m) / bw,
                                    loc=m,
                                    scale=bw,
                                    random_state=self.random_state,
                                )
                            )
                        else:
                            # categorical or integer parameter
                            if self.random_state.rand() < (1 - bw):
                                candidate.append(m)
                            else:
                                if vartype == "o":
                                    # integer
                                    sample = self.random_state.randint(
                                        domain[0], domain[1]
                                    )
                                    sample = (sample - domain[0]) / (
                                        domain[1] - domain[0]
                                    )
                                    candidate.append(sample)
                                elif vartype == "u":
                                    # categorical
                                    candidate.append(
                                        self.random_state.randint(domain) / domain
                                    )
                    val = acquisition_function(candidate)

                    if not np.isfinite(val):
                        logging.warning(
                            "candidate has non finite acquisition function value"
                        )

                    if val_current_best is None or val_current_best > val:
                        current_best = candidate
                        val_current_best = val

                suggestion = self.from_feature(feature_vector=current_best)

        return suggestion

    def train_kde(self, train_data, train_targets):

        if train_data.shape[0] < self.num_min_data_points:
            return None

        n_good = max(
            self.num_min_data_points, (self.top_n_percent * train_data.shape[0]) // 100
        )
        n_bad = max(
            self.num_min_data_points,
            ((100 - self.top_n_percent) * train_data.shape[0]) // 100,
        )

        idx = np.argsort(train_targets)

        train_data_good = train_data[idx[:n_good]]
        train_data_bad = train_data[idx[n_good : n_good + n_bad]]

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return None
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return None

        types = [t[0] for t in self.vartypes]

        bad_kde = sm.nonparametric.KDEMultivariate(
            data=train_data_bad, var_type=types, bw="normal_reference"
        )
        good_kde = sm.nonparametric.KDEMultivariate(
            data=train_data_good, var_type=types, bw="normal_reference"
        )

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        return bad_kde, good_kde

    def clone_from_state(self, state: dict):
        raise NotImplementedError
