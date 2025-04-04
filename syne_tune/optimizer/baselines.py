from typing import Dict, Optional, Any, List
import logging

from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune.optimizer.schedulers.searchers.botorch.botorch_searcher import (
    BoTorchSearcher,
)
from syne_tune.optimizer.schedulers.searchers.conformal.surrogate_searcher import (
    SurrogateSearcher,
)
from syne_tune.optimizer.schedulers.searchers.random_searcher import RandomSearcher
from syne_tune.optimizer.schedulers.searchers.bore import Bore
from syne_tune.optimizer.schedulers.searchers.kde import (
    KernelDensityEstimator,
)
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)
from syne_tune.optimizer.schedulers.single_fidelity_scheduler import (
    SingleFidelityScheduler,
)
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

logger = logging.getLogger(__name__)


class RandomSearch(SingleFidelityScheduler):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metrics: List[str],
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(RandomSearch, self).__init__(
            config_space=config_space,
            metrics=metrics,
            do_minimize=do_minimize,
            searcher=RandomSearcher(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
            ),
            random_seed=random_seed,
        )


class BORE(SingleObjectiveScheduler):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(BORE, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            searcher=Bore(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
            ),
            random_seed=random_seed,
        )


class TPE(SingleObjectiveScheduler):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
        num_min_data_points: Optional[int] = None,
        top_n_percent: int = 15,
        min_bandwidth: float = 1e-3,
        num_candidates: int = 64,
        bandwidth_factor: int = 3,
        random_fraction: float = 0.33,
    ):
        super(TPE, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            searcher=KernelDensityEstimator(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
                num_min_data_points=num_min_data_points,
                top_n_percent=top_n_percent,
                min_bandwidth=min_bandwidth,
                num_candidates=num_candidates,
                bandwidth_factor=bandwidth_factor,
                random_fraction=random_fraction,
            ),
            random_seed=random_seed,
        )


class REA(SingleObjectiveScheduler):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        population_size: int = 100,
        sample_size: int = 10,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(REA, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            searcher=RegularizedEvolution(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
                population_size=population_size,
                sample_size=sample_size,
            ),
            random_seed=random_seed,
        )


class BOTorch(SingleObjectiveScheduler):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(BOTorch, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            searcher=BoTorchSearcher(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
            ),
            random_seed=random_seed,
        )


class ASHA(AsynchronousSuccessiveHalving):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        time_attr: str,
        max_t: int,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(ASHA, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            max_t=max_t,
            searcher=RandomSearcher(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
            ),
            random_seed=random_seed,
            time_attr=time_attr,
        )


class ASHABORE(AsynchronousSuccessiveHalving):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        time_attr: str,
        max_t: int,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(ASHABORE, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            max_t=max_t,
            searcher="bore",
            random_seed=random_seed,
            time_attr=time_attr,
            searcher_kwargs={"points_to_evaluate": points_to_evaluate},
        )


class ASHACQR(AsynchronousSuccessiveHalving):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        time_attr: str,
        max_t: int,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(ASHACQR, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            max_t=max_t,
            searcher="cqr",
            random_seed=random_seed,
            time_attr=time_attr,
            searcher_kwargs={"points_to_evaluate": points_to_evaluate},
        )


class BOHB(AsynchronousSuccessiveHalving):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        time_attr: str,
        max_t: int,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        num_min_data_points: Optional[int] = None,
        top_n_percent: int = 15,
        min_bandwidth: float = 1e-3,
        num_candidates: int = 64,
        bandwidth_factor: int = 3,
        random_fraction: float = 0.33,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(BOHB, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            max_t=max_t,
            searcher="kde",
            searcher_kwargs={
                "config_space": config_space,
                "points_to_evaluate": points_to_evaluate,
                "random_seed": random_seed,
                "num_min_data_points": num_min_data_points,
                "top_n_percent": top_n_percent,
                "min_bandwidth": min_bandwidth,
                "num_candidates": num_candidates,
                "bandwidth_factor": bandwidth_factor,
                "random_fraction": random_fraction,
            },
            random_seed=random_seed,
            time_attr=time_attr,
        )


class CQR(SingleObjectiveScheduler):
    """Random search.

    See :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    for ``kwargs["search_options"]`` parameters.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        do_minimize: Optional[bool] = True,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[dict]] = None,
    ):
        super(CQR, self).__init__(
            config_space=config_space,
            metric=metric,
            do_minimize=do_minimize,
            searcher=SurrogateSearcher(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
            ),
            random_seed=random_seed,
        )


# Dictionary that allows to also list baselines who don't need a wrapper class
# such as :class:`PopulationBasedTraining`
baselines_dict = {
    "Random Search": RandomSearch,
    "BORE": BORE,
    "TPE": TPE,
    "REA": REA,
    "BOTorch": BOTorch,
}
