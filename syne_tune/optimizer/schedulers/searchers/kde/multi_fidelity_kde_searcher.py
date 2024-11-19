from typing import Dict, Optional, List, Any, Tuple
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers.kde.kde_searcher import (
    KernelDensityEstimator,
)

logger = logging.getLogger(__name__)


class MultiFidelityKernelDensityEstimator(KernelDensityEstimator):
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
        random_seed: int | None = None,
    ):
        super().__init__(
            config_space,
            points_to_evaluate=points_to_evaluate,
            num_min_data_points=num_min_data_points,
            top_n_percent=top_n_percent,
            min_bandwidth=min_bandwidth,
            num_candidates=num_candidates,
            bandwidth_factor=bandwidth_factor,
            random_fraction=random_fraction,
            random_seed=random_seed,
        )
        self.resource_levels = []

    def _highest_resource_model_can_fit(self, num_features: int) -> Optional[int]:
        unique_resource_levels, counts = np.unique(
            self.resource_levels, return_counts=True
        )
        for resource, count in reversed(list(zip(unique_resource_levels, counts))):
            if self._check_data_shape_and_good_size((count, num_features)) is not None:
                return resource
        return None

    def _train_kde(
        self, train_data: np.ndarray, train_targets: np.ndarray
    ) -> Optional[Tuple[Any, Any]]:
        """
        Find the highest resource level so that the data only at that level is
        large enough to train KDE models both on the good part and the rest.
        If no such resource level exists, we return ``None``.

        :param train_data: Training input features
        :param train_targets: Training targets
        :return: Tuple of good model, bad model; or ``None``
        """
        train_data = train_data.reshape((train_targets.size, -1))
        num_features = train_data.shape[1]
        resource = self._highest_resource_model_can_fit(num_features)
        if resource is None:
            return None
        else:
            # Models can be fit
            indices = np.where(self.resource_levels == resource)
            sub_data = train_data[indices]
            sub_targets = train_targets[indices]
            return super()._train_kde(sub_data, sub_targets)

    def on_trial_result(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: float,
    ):
        super().on_trial_result(trial_id=trial_id, config=config, metric=metric)
        resource_level = int(resource_level)
        self.resource_levels.append(resource_level)
