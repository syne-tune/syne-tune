import logging
import numpy as np

from copy import deepcopy
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class BaseSearcher:
    """
    Base class for searchers that sample hyperparameter configurations
    from the given configuration space.

    :param config_space: The configuration space to sample from.
    :param points_to_evaluate: A list of configurations to evaluate initially (in the given order).
    :param random_seed: Seed used to initialize the random number generators.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        random_seed: int = None,
    ):
        self.config_space = config_space
        if points_to_evaluate is None:
            self.points_to_evaluate = []
        else:
            self.points_to_evaluate = deepcopy(points_to_evaluate)

        if random_seed is None:
            self.random_seed = np.random.randint(0, 2**31 - 1)
        else:
            self.random_seed = random_seed

    def _next_points_to_evaluate(self) -> Optional[Dict[str, Any]]:
        """
        :return: Next entry from remaining ``points_to_evaluate`` (popped
            from front), or None
        """
        if self.points_to_evaluate:
            return self.points_to_evaluate.pop(0)
        else:
            return None  # No more initial configs

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
        raise NotImplementedError

    def on_trial_result(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metrics: List[float],
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
        :param metrics: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """
        return

    def on_trial_error(self, trial_id: int):
        """Called by scheduler if an evaluation job for a trial failed.

        The searcher should react appropriately (e.g., remove pending evaluations
        for this trial, not suggest the configuration again).

        :param trial_id: ID of trial whose evaluated failed
        """
        return

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metrics: List[float],
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
        :param metrics: See :meth:`~syne_tune.optimizer.schedulers.TrialScheduler.on_trial_result`
        """
        return
