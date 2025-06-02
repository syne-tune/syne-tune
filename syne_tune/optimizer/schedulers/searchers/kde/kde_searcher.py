from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np
import statsmodels.api as sm
import scipy.stats as sps

from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
import syne_tune.config_space as sp

logger = logging.getLogger(__name__)


class KernelDensityEstimator(SingleObjectiveBaseSearcher):
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

        | Algorithms for Hyper-Parameter Optimization
        | J. Bergstra and R. Bardenet and Y. Bengio and B. K{\'e}gl
        | Proceedings of the 24th International Conference on Advances in Neural Information Processing Systems
        | https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html

    and

        | BOHB: Robust and Efficient Hyperparameter Optimization at Scale
        | S. Falkner and A. Klein and F. Hutter
        | Proceedings of the 35th International Conference on Machine Learning
        | https://arxiv.org/abs/1807.01774

    :param config_space: Configuration space for the evaluation function.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
    :param num_min_data_points: Minimum number of data points that we use to fit
        the KDEs. As long as less observations have been received in
        :meth:`update`, randomly drawn configurations are returned in
        :meth:`get_config`.
        If set to ``None``, we set this to the number of hyperparameters.
        Defaults to ``None``.
    :param top_n_percent: Determines how many datapoints we use to fit the first
        KDE model for modeling the well performing configurations.
        Defaults to 15
    :param min_bandwidth: The minimum bandwidth for the KDE models. Defaults
        to 1e-3
    :param num_candidates: Number of candidates that are sampled to optimize
        the acquisition function. Defaults to 64
    :param bandwidth_factor: We sample continuous hyperparameter from a
        truncated Normal. This factor is multiplied to the bandwidth to define
        the standard deviation of this truncated Normal. Defaults to 3
    :param random_fraction: Defines the fraction of configurations that are
        drawn uniformly at random instead of sampling from the model.
        Defaults to 0.33
    :param random_seed: Seed for initializing random number generators.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        points_to_evaluate: Optional[List[dict]] = None,
        num_min_data_points: Optional[int] = None,
        top_n_percent: int = 15,
        min_bandwidth: float = 1e-3,
        num_candidates: int = 64,
        bandwidth_factor: int = 3,
        random_fraction: float = 0.33,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            config_space=config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )

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
        self.random_state = np.random.RandomState(self.random_seed)

        self.good_kde = None
        self.bad_kde = None

        self.vartypes = list()

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
        assert self.num_min_data_points >= len(
            self.vartypes
        ), f"num_min_data_points = {num_min_data_points}, must be >= {len(self.vartypes)}"

    def _to_feature(self, config):
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

    def _from_feature(self, feature_vector):
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

        res = dict()
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

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
    ):
        self.X.append(self._to_feature(config=config))
        self.y.append(metric)

    def _get_random_config(self):
        return {
            k: v.sample() if hasattr(v, "sample") else v
            for k, v in self.config_space.items()
        }

    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        suggestion = self._next_points_to_evaluate()
        if suggestion is None:
            if self.y:
                models = self._train_kde(np.array(self.X), np.array(self.y))
            else:
                models = None

            if models is None or self.random_state.rand() < self.random_fraction:
                # return random candidate because a) we don't have enough data points or
                # b) we sample some fraction of all samples randomly
                suggestion = self._get_random_config()
            else:
                self.bad_kde = models[0]
                self.good_kde = models[1]
                l = self.good_kde.pdf
                g = self.bad_kde.pdf

                def acquisition_function(x):
                    return max(1e-32, g(x)) / max(l(x), 1e-32)

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

                    config = self._from_feature(candidate)
                    if val_current_best is None or val_current_best > val:
                        suggestion = config
                        val_current_best = val

                if suggestion is None:
                    # This can happen if the configuration space is almost exhausted
                    logger.warning(
                        "Could not find configuration by optimizing the acquisition function. Drawing at random instead."
                    )
                    suggestion = self._get_random_config()

        return suggestion

    def _check_data_shape_and_good_size(
        self, data_shape: Tuple[int, int]
    ) -> Optional[int]:
        """
        Determine size of data for "good" model (the rest of the data is for the
        "bad" model). Both sizes must be larger than the number of features,
        otherwise ``None`` is returned.

        :param data_shape: Shape of ``train_data``
        :return: Size of data for "good" model, or ``None`` (models cannot be
            fit, too little data)
        """
        num_data, num_features = data_shape
        n_good = max(self.num_min_data_points, (self.top_n_percent * num_data) // 100)
        # Number of data points have to be larger than the number of features to meet
        # the input constraints of ``statsmodels.KDEMultivariate``
        if min(n_good, num_data - n_good) <= num_features:
            return None
        else:
            return n_good

    def _train_kde(
        self, train_data: np.ndarray, train_targets: np.ndarray
    ) -> Optional[Tuple[Any, Any]]:
        train_data = train_data.reshape((train_targets.size, -1))
        n_good = self._check_data_shape_and_good_size(train_data.shape)
        if n_good is None:
            return None

        idx = np.argsort(train_targets)
        train_data_good = train_data[idx[:n_good]]
        train_data_bad = train_data[idx[n_good:]]

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
