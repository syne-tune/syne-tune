import logging
import numpy as np

from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)


class BaseSearcher:
    """
    Base class of searchers, which are components of schedulers responsible for
    implementing :meth:`get_config`.

    # TODO: Update docstrings
    .. note::
       This is an abstract base class. In order to implement a new searcher, try to
       start from
       :class:`~syne_tune.optimizer.scheduler.searcher.StochasticAndFilterDuplicatesSearcher`
       or :class:`~syne_tune.optimizer.scheduler.searcher.StochasticSearcher`,
       which implement generally useful properties.

    :param config_space: Configuration space
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If ``None`` (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
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
            self.points_to_evaluate = points_to_evaluate

        if random_seed is None:
            self.random_seed = np.random.randint(0, 2**31 - 1)
        else:
            self.random_seed = random_seed

    def _next_initial_config(self) -> Optional[Dict[str, Any]]:
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

        Note: Query :meth:`_next_initial_config` for initial configs to return
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
        metric: List[float],
        update: bool,
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
        :param update: Should surrogate model be updated?
        """
        return

    def on_trial_error(self, trial_id: int):
        """Called by scheduler if an evaluation job for a trial failed.

        The searcher should react appropriately (e.g., remove pending evaluations
        for this trial, not suggest the configuration again).

        :param trial_id: ID of trial whose evaluated failed
        """
        return
