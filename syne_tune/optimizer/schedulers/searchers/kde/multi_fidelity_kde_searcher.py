from typing import Dict, Optional, List, Any
import logging
from collections import OrderedDict

from syne_tune.optimizer.schedulers.searchers.kde.kde_searcher import (
    KernelDensityEstimator,
)
from syne_tune.optimizer.schedulers.searchers.multi_fidelity_searcher import (
    MultiFidelityBaseSearcher,
)

logger = logging.getLogger(__name__)


class MultiFidelityKernelDensityEstimator(MultiFidelityBaseSearcher):
    """
    Adapts :class:`KernelDensityEstimator` to the multi-fidelity setting as proposed
    by Falkner et al such that we can use it with Hyperband. Following Falkner
    et al, we fit the KDE only on the highest resource level where we have at
    least num_min_data_points. Code is based on the implementation by Falkner
    et al: https://github.com/automl/HpBandSter/tree/master/hpbandster

        | BOHB: Robust and Efficient Hyperparameter Optimization at Scale
        | S. Falkner and A. Klein and F. Hutter
        | Proceedings of the 35th International Conference on Machine Learning

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.kde.KernelDensityEstimator`:
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
            config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )

        self.num_min_data_points = num_min_data_points
        self.min_bandwidth = min_bandwidth
        self.random_fraction = random_fraction
        self.num_candidates = num_candidates
        self.bandwidth_factor = bandwidth_factor
        self.top_n_percent = top_n_percent

        self.models = OrderedDict()

    def initialize_model(self):
        return KernelDensityEstimator(
            config_space=self.config_space,
            num_min_data_points=self.num_min_data_points,
            min_bandwidth=self.min_bandwidth,
            random_fraction=self.random_fraction,
            num_candidates=self.num_candidates,
            bandwidth_factor=self.bandwidth_factor,
            top_n_percent=self.top_n_percent,
            random_seed=self.random_seed,
        )

    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Suggest a new configuration.

        Note: Query :meth:`_next_points_to_evaluate` for initial configs to return
        first.

        :param kwargs: Extra information may be passed from scheduler to
            searcher
        :return: New configuration. The searcher may return None if a new
            configuration cannot be suggested. In this case, the tuning will
            stop. This happens if searchers never suggest the same config more
            than once, and all configs in the (finite) search space are
            exhausted.
        """
        suggestion = self._next_points_to_evaluate()
        if suggestion is not None:
            return suggestion

        if len(self.models) == 0:
            return {
                k: v.sample() if hasattr(v, "sample") else v
                for k, v in self.config_space.items()
            }

        highest_observed_resource = next(reversed(self.models))
        return self.models[highest_observed_resource].suggest()

    def on_trial_result(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int,
    ):
        """Inform searcher about result

        The scheduler passes every result. If ``update == True``, the searcher
        should update its surrogate model (if any), otherwise ``result`` is an
        intermediate result not modelled.

        The default implementation calls :meth:`_update` if ``update == True``.
        It can be overwritten by searchers which also react to intermediate
        results.

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """
        if resource_level not in self.models:
            self.models[resource_level] = self.initialize_model()

        self.models[resource_level].on_trial_complete(
            trial_id=trial_id, config=config, metric=metric
        )

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int,
    ):
        """Inform searcher about result

        The scheduler passes every result. If ``update == True``, the searcher
        should update its surrogate model (if any), otherwise ``result`` is an
        intermediate result not modelled.

        The default implementation calls :meth:`_update` if ``update == True``.
        It can be overwritten by searchers which also react to intermediate
        results.

        :param trial_id: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param config: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        :param metric: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """

        if resource_level not in self.models:
            self.models[resource_level] = self.initialize_model()

        self.models[resource_level].on_trial_complete(
            trial_id=trial_id, config=config, metric=metric
        )
