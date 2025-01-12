from typing import Optional, List, Dict, Any
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers.bore import Bore

logger = logging.getLogger(__name__)


class MultiFidelityBore(Bore):
    """
    Adapts BORE (Tiao et al.) for the multi-fidelity Hyperband setting following
    BOHB (Falkner et al.). Once we collected enough data points on the smallest
    resource level, we fit a probabilistic classifier and sample from it until we have
    a sufficient amount of data points for the next higher resource level. We then
    refit the classifier on the data of this resource level. These steps are
    iterated until we reach the highest resource level. References:

        | BORE: Bayesian Optimization by Density-Ratio Estimation,
        | Tiao, Louis C and Klein, Aaron and Seeger, Matthias W and Bonilla, Edwin V. and Archambeau, Cedric and Ramos, Fabio
        | Proceedings of the 38th International Conference on Machine Learning

    and

        | BOHB: Robust and Efficient Hyperparameter Optimization at Scale
        | S. Falkner and A. Klein and F. Hutter
        | Proceedings of the 35th International Conference on Machine Learning

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.bore.Bore`:

    :param resource_attr: Name of resource attribute. Defaults to "epoch"
    """

    def __init__(
            self,
            resource_attr: str,
            config_space: Dict[str, Any],
            points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
            random_seed: int = None,
            gamma: Optional[float] = 0.25,
            calibrate: Optional[bool] = False,
            classifier: Optional[str] = "xgboost",
            acq_optimizer: Optional[str] = "rs",
            feval_acq: Optional[int] = 500,
            random_prob: Optional[float] = 0.0,
            init_random: Optional[int] = 6,
            classifier_kwargs: Optional[dict] = None,
    ):
        if acq_optimizer is None:
            acq_optimizer = "rs_with_replacement"
        super().__init__(
            config_space,
            points_to_evaluate=points_to_evaluate,
            gamma=gamma,
            calibrate=calibrate,
            classifier=classifier,
            acq_optimizer=acq_optimizer,
            feval_acq=feval_acq,
            random_prob=random_prob,
            init_random=init_random,
            classifier_kwargs=classifier_kwargs,
            random_seed=random_seed
        )
        self.resource_attr = resource_attr
        self.resource_levels = []

    def _train_model(self, train_data: np.ndarray, train_targets: np.ndarray) -> bool:
        # find the highest resource level we have at least one data points of the positive class
        min_data_points = int(1 / self.gamma)
        unique_resource_levels, counts = np.unique(
            self.resource_levels, return_counts=True
        )
        idx = np.where(counts >= min_data_points)[0]

        if len(idx) == 0:
            return False

        # collect data on the highest resource level
        highest_resource_level = unique_resource_levels[idx[-1]]
        indices = np.where(self.resource_levels == highest_resource_level)[0]

        train_data = np.array([self.inputs[i] for i in indices])
        train_targets = np.array([self.targets[i] for i in indices])

        return super()._train_model(train_data, train_targets)


    def on_trial_result(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int = None,
    ):
        self.resource_levels.append(resource_level)
        self.inputs.append(self._hp_ranges.to_ndarray(config))
        self.targets.append(metric)
