from typing import Dict

from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from syne_tune.optimizer.schedulers.pbt import PopulationBasedTraining
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import RegularizedEvolution


class RandomSearch(FIFOScheduler):
    def __init__(self, config_space: Dict, metric: str, **kwargs):
        super(RandomSearch, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            **kwargs,
        )


class BayesianOptimization(FIFOScheduler):
    def __init__(self, config_space: Dict, metric: str, **kwargs):
        super(BayesianOptimization, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="bayesopt",
            **kwargs,
        )


class ASHA(HyperbandScheduler):
    def __init__(self, config_space: Dict, metric: str, resource_attr: str, max_t: int, **kwargs):
        super(ASHA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            resource_attr=resource_attr,
            max_t=max_t,
            **kwargs,
        )


class MOBSTER(HyperbandScheduler):
    def __init__(self, config_space: Dict, metric: str, resource_attr: str, max_t: int, **kwargs):
        super(MOBSTER, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="bayesopt",
            resource_attr=resource_attr,
            max_t=max_t,
            **kwargs,
        )


class PASHA(HyperbandScheduler):
    def __init__(
            self,
            config_space: Dict,
            metric: str,
            resource_attr: str,
            max_t: int,
            **kwargs,
    ):
        super(PASHA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            resource_attr=resource_attr,
            max_t=max_t,
            type='pasha',
            ranking_criterion='soft_ranking',
            **kwargs,
        )


class BORE(FIFOScheduler):
    def __init__(self, config_space: Dict, metric: str, mode: str, **kwargs):
        from syne_tune.optimizer.schedulers.searchers.bore import Bore
        super(BORE, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=Bore(config_space=config_space, metric=metric, mode=mode),
            mode=mode,
            **kwargs,
        )


class REA(FIFOScheduler):
    def __init__(self, config_space: Dict, metric: str, population_size: int = 100, sample_size: int = 10, **kwargs):
        super(REA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=RegularizedEvolution(configspace=config_space, metric=metric,
                                          population_size=population_size, sample_size=sample_size,
                                          **kwargs),
            **kwargs,
        )


# dictionary that allows to also list baselines who don't need a wrapper class such as PBT.
baselines = {
    'Random Search': RandomSearch,
    'Bayesian Optimization': BayesianOptimization,
    'ASHA': ASHA,
    'MOBSTER': MOBSTER,
    'PASHA': PASHA,
    'MOASHA': MOASHA,
    'PBT': PopulationBasedTraining,
    'BORE': BORE,
}
