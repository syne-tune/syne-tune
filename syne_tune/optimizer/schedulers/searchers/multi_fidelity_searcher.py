import logging

from typing import Dict, Any, Optional

from syne_tune.optimizer.schedulers.searchers.searcher import BaseSearcher

logger = logging.getLogger(__name__)


class MultiFidelityBaseSearcher(BaseSearcher):
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
        return

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
        return
