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
    Base class to implement scheduler that optimize a single objective.

    :param config_space: Configuration space for evaluation function
    :param searcher: Searcher for ``get_config`` decisions. String values
        are passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory` along
        with ``search_options`` and extra information. Supported values:
        :const:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.SUPPORTED_SEARCHERS_FIFO`.
        Defaults to "random" (i.e., random search)
    :param metric: Name of metric to optimize, key in results obtained via
        ``on_trial_result``.
    :param do_minimize: True if we minimize the objective function
    :param random_seed: Seed used to initialize the random number generators.
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
