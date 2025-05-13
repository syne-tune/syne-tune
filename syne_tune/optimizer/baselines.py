from typing import Dict, Optional, Any, List
import logging

from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune.optimizer.schedulers.searchers.botorch.botorch_searcher import (
    BoTorchSearcher,
)
from syne_tune.optimizer.schedulers.searchers.conformal.conformal_quantile_regression_searcher import (
    ConformalQuantileRegression,
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
    """
    Random search that samples hyperparameter configurations uniformly at random in each iteration.
    Supports both single- and multi-objective optimization.

    :param config_space: Configuration space for the evaluation function.
    :param metrics: Name(s) of the metric(s) to optimize.
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
    """
    Bayesian Optimization by Density-Ratio Estimation (BORE) as proposed by:

        | BORE: Bayesian Optimization by Density-Ratio Estimation,
        | Tiao, Louis C and Klein, Aaron and Seeger, Matthias W and Bonilla, Edwin V. and Archambeau, Cedric and Ramos, Fabio
        | Proceedings of the 38th International Conference on Machine Learning
        | https://arxiv.org/abs/2102.09009

    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of the metric to optimize.
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
    """
    Tree-Parzen Estimator as proposed by:

        | Algorithms for Hyper-Parameter Optimization
        | J. Bergstra and R. Bardenet and Y. Bengio and B. K{\'e}gl
        | Proceedings of the 24th International Conference on Advances in Neural Information Processing Systems
        | https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html


    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of the metric to optimize.
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
    """
    Regularized Evolution as proposed by

        | Real, E., Aggarwal, A., Huang, Y., and Le, Q. V.
        | Regularized Evolution for Image Classifier Architecture Search.
        | In Proceedings of the Conference on Artificial Intelligence (AAAI’19)

    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of the metric to optimize.
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param population_size: Size of the population, defaults to 100
    :param sample_size: Size of the candidate set to obtain a parent for the
        mutation, defaults to 10
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
    """
    Implements Gaussian-process based Bayesian optimization based on BOTorch.

    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of the metric to optimize.
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
    """
    Asynchronous Successive Halving (ASHA) as proposed by:

        | Li, Jamieson, Rostamizadeh, Gonina, Hardt, Recht, Talwalkar (2018)
        | A System for Massively Parallel Hyperparameter Tuning
        | https://arxiv.org/abs/1810.05934


    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of the metric to optimize.
    :param time_attr: A training result attr to use for comparing time.
        Note that you can pass in something non-temporal such as
        ``training_iteration`` as a measure of progress, the only requirement
        is that the attribute should increase monotonically.
        Defaults to "training_iteration"
    :param max_t: max time units per trial. Trials will be stopped after
        ``max_t`` time units (determined by ``time_attr``) have passed.
        Defaults to 100
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
            searcher="random_search",
            random_seed=random_seed,
            time_attr=time_attr,
        )


class ASHABORE(AsynchronousSuccessiveHalving):
    """
    Asynchronous Successive Halving (ASHA) which fits a BORE model on each rung level.

    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of the metric to optimize.
    :param time_attr: A training result attr to use for comparing time.
        Note that you can pass in something non-temporal such as
        ``training_iteration`` as a measure of progress, the only requirement
        is that the attribute should increase monotonically.
        Defaults to "training_iteration"
    :param max_t: max time units per trial. Trials will be stopped after
        ``max_t`` time units (determined by ``time_attr``) have passed.
        Defaults to 100
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
    """
    Asynchronous Successive Halving (ASHA) which fits a CQR model on each rung level.

    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of the metric to optimize.
    :param time_attr: A training result attr to use for comparing time.
        Note that you can pass in something non-temporal such as
        ``training_iteration`` as a measure of progress, the only requirement
        is that the attribute should increase monotonically.
        Defaults to "training_iteration"
    :param max_t: max time units per trial. Trials will be stopped after
        ``max_t`` time units (determined by ``time_attr``) have passed.
        Defaults to 100
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
    """
    Bayesian Optimization Hyperband combines ASHA with TPE-like Bayesian optimization, using kernel
    density estimators as proposed by:

        | BOHB: Robust and Efficient Hyperparameter Optimization at Scale
        | S. Falkner and A. Klein and F. Hutter
        | Proceedings of the 35th International Conference on Machine Learning
        | https://arxiv.org/abs/1807.01774

    Compared to the method proposed by Falkner et al. we use asynchronous successive halving for scheduling.

    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of the metric to optimize.
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
    """
    Single-fidelity Conformal Quantile Regression approach proposed in:
        | Optimizing Hyperparameters with Conformal Quantile Regression.
        | David Salinas, Jacek Golebiowski, Aaron Klein, Matthias Seeger, Cedric Archambeau.
        | ICML 2023.
    The method predict quantile performance with gradient boosted trees and calibrate prediction with conformal
    predictions.

    :param config_space: Configuration space for the evaluation function.
    :param metric: Name of the metric to optimize.
    :param do_minimize: Set to True if the objective function should be minimized.
    :param random_seed: Seed for initializing random number generators.
    :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
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
            searcher=ConformalQuantileRegression(
                config_space=config_space,
                points_to_evaluate=points_to_evaluate,
                random_seed=random_seed,
            ),
            random_seed=random_seed,
        )


baselines_dict = {
    "Random Search": RandomSearch,
    "BORE": BORE,
    "TPE": TPE,
    "REA": REA,
    "BOTorch": BOTorch,
}
