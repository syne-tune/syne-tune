from typing import Optional, Dict, Any, Union
import logging

from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from syne_tune.optimizer.schedulers.single_fidelity_scheduler import (
    SingleFidelityScheduler,
)

logger = logging.getLogger(__name__)


class SingleObjectiveScheduler(SingleFidelityScheduler):
    """
    #TODO: Update docstring

    Schedulers maintain and drive the logic of an experiment, making decisions
    which configs to evaluate in new trials, and which trials to stop early.

    Some schedulers support pausing and resuming trials. In this case, they
    also drive the decision when to restart a paused trial.

    :param config_space: Configuration space for evaluation function
    :type config_space: Dict[str, Any]
    :param searcher: Searcher for ``get_config`` decisions. String values
        are passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory` along
        with ``search_options`` and extra information. Supported values:
        :const:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.SUPPORTED_SEARCHERS_FIFO`.
        Defaults to "random" (i.e., random search)
    :type searcher: str or
        :class:`~syne_tune.optimizer.schedulers.searchers.BaseSearcher`
    :param metric: Name of metric to optimize, key in results obtained via
        ``on_trial_result``.
    :type metric: str
    :param random_seed: Master random seed. Generators used in the
        scheduler or searcher are seeded using :class:`RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        searcher: Optional[Union[str, SingleObjectiveBaseSearcher]] = "random_search",
        random_seed: int = None,
        searcher_kwargs: dict = None,
    ):

        super().__init__(
            random_seed=random_seed,
            config_space=config_space,
            do_minimize=do_minimize,
            searcher=searcher,
            metrics=[metric],
            searcher_kwargs=searcher_kwargs,
        )
