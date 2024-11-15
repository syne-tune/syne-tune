from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np
import statsmodels.api as sm
import scipy.stats as sps

from syne_tune.optimizer.schedulers.searchers import (
    StochasticAndFilterDuplicatesSearcher,
)
import syne_tune.config_space as sp
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)

logger = logging.getLogger(__name__)


class LegacyKernelDensityEstimator(StochasticAndFilterDuplicatesSearcher):
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

    Note: ``restrict_configurations`` is not supported here, this would require
    reimplementing the selection of configs in :meth:`_get_config`.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.StochasticAndFilterDuplicatesSearcher`:

    :param mode: Mode to use for the metric given, can be "min" or "max". Is
        obtained from scheduler in :meth:`configure_scheduler`. Defaults to "min"
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
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        allow_duplicates: Optional[bool] = None,
        mode: Optional[str] = None,
        num_min_data_points: Optional[int] = None,
        top_n_percent: Optional[int] = None,
        min_bandwidth: Optional[float] = None,
        num_candidates: Optional[int] = None,
        bandwidth_factor: Optional[int] = None,
        random_fraction: Optional[float] = None,
        **kwargs,
    ):
        k = "restrict_configurations"
        if kwargs.get(k) is not None:
            logger.warning(f"{k} is not supported")
            del kwargs[k]
        super().__init__(
            config_space=config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            allow_duplicates=allow_duplicates,
            mode="min" if mode is None else mode,
            **kwargs,
        )
        if top_n_percent is None:
            top_n_percent = 15
        if min_bandwidth is None:
            min_bandwidth = 1e-3
        if num_candidates is None:
            num_candidates = 64
        if bandwidth_factor is None:
            bandwidth_factor = 3
        if random_fraction is None:
            random_fraction = 0.33
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
        assert self.num_min_data_points >= len(
            self.vartypes
        ), f"num_min_data_points = {num_min_data_points}, must be >= {len(self.vartypes)}"
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

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.scheduler_searcher import (
            TrialSchedulerWithSearcher,
        )

        assert isinstance(
            scheduler, TrialSchedulerWithSearcher
        ), "This searcher requires TrialSchedulerWithSearcher scheduler"
        super().configure_scheduler(scheduler)

    def _to_objective(self, result: Dict[str, Any]) -> float:
        if self._mode == "min":
            return result[self._metric]
        else:
            return -result[self._metric]

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        self.X.append(self._to_feature(config=config))
        self.y.append(self._to_objective(result))
        if self._debug_log is not None:
            metric_val = result[self._metric]
            if self._resource_attr is not None:
                # For HyperbandScheduler, also add the resource attribute
                resource = int(result[self._resource_attr])
                trial_id = trial_id + ":{}".format(resource)
            msg = f"Update for trial_id {trial_id}: metric = {metric_val:.3f}"
            logger.info(msg)

    def _get_config(self, **kwargs) -> Optional[Dict[str, Any]]:
        suggestion = self._next_initial_config()
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
                    if (
                        val_current_best is None or val_current_best > val
                    ) and not self.should_not_suggest(config):
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

    def clone_from_state(self, state: Dict[str, Any]):
        raise NotImplementedError
