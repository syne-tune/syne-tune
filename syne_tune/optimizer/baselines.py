from typing import Dict

from syne_tune.optimizer.schedulers import FIFOScheduler, HyperbandScheduler, \
    PopulationBasedTraining
from syne_tune.optimizer.schedulers.multiobjective import MOASHA
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import \
    RegularizedEvolution
from syne_tune.optimizer.schedulers.synchronous import \
    SynchronousGeometricHyperbandScheduler
from syne_tune.optimizer.schedulers.median_stopping_rule import \
    MedianStoppingRule


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
    def __init__(
            self, config_space: Dict, metric: str, resource_attr: str,
            **kwargs):
        """
        One of `max_t`, `max_resource_attr` needs to be in `kwargs`. For
        `type='promotion'`, the latter is more useful, see also
        :class:`HyperbandScheduler`.

        """
        need_one = {'max_t', 'max_resource_attr'}
        assert need_one.intersection(kwargs.keys()), \
            f"Need one of these: {need_one}"
        super(ASHA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            resource_attr=resource_attr,
            **kwargs,
        )


class MOBSTER(HyperbandScheduler):
    def __init__(
            self, config_space: Dict, metric: str, resource_attr: str,
            **kwargs):
        """
        One of `max_t`, `max_resource_attr` needs to be in `kwargs`. For
        `type='promotion'`, the latter is more useful, see also
        :class:`HyperbandScheduler`.

        """
        need_one = {'max_t', 'max_resource_attr'}
        assert need_one.intersection(kwargs.keys()), \
            f"Need one of these: {need_one}"
        super(MOBSTER, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="bayesopt",
            resource_attr=resource_attr,
            **kwargs,
        )


class PASHA(HyperbandScheduler):
    def __init__(
            self, config_space: Dict, metric: str, resource_attr: str,
            **kwargs):
        """
        One of `max_t`, `max_resource_attr` needs to be in `kwargs`. The
        latter is more useful, see also :class:`HyperbandScheduler`.

        """
        need_one = {'max_t', 'max_resource_attr'}
        assert need_one.intersection(kwargs.keys()), \
            f"Need one of these: {need_one}"
        super(PASHA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            resource_attr=resource_attr,
            type='pasha',
            **kwargs,
        )


class SyncHyperband(SynchronousGeometricHyperbandScheduler):
    def __init__(
            self, config_space: Dict, metric: str, resource_attr: str,
            **kwargs):
        """
        One of `max_resource_level`, `max_resource_attr` needs to be in
        `kwargs`. The latter is more useful, see also
        :class:`HyperbandScheduler`.

        """
        need_one = {'max_resource_level', 'max_resource_attr'}
        assert need_one.intersection(kwargs.keys()), \
            f"Need one of these: {need_one}"
        super(SyncHyperband, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            resource_attr=resource_attr,
            **kwargs,
        )


class SyncBOHB(SynchronousGeometricHyperbandScheduler):
    def __init__(
            self, config_space: Dict, metric: str, resource_attr: str,
            **kwargs):
        """
        One of `max_resource_level`, `max_resource_attr` needs to be in
        `kwargs`. The latter is more useful, see also
        :class:`HyperbandScheduler`.

        """
        need_one = {'max_resource_level', 'max_resource_attr'}
        assert need_one.intersection(kwargs.keys()), \
            f"Need one of these: {need_one}"
        super(SyncBOHB, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="kde",
            resource_attr=resource_attr,
            **kwargs,
        )


class SyncMOBSTER(SynchronousGeometricHyperbandScheduler):
    def __init__(
            self, config_space: Dict, metric: str, resource_attr: str,
            **kwargs):
        """
        One of `max_resource_level`, `max_resource_attr` needs to be in
        `kwargs`. The latter is more useful, see also
        :class:`HyperbandScheduler`.

        """
        need_one = {'max_resource_level', 'max_resource_attr'}
        assert need_one.intersection(kwargs.keys()), \
            f"Need one of these: {need_one}"
        super(SyncMOBSTER, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="bayesopt",
            resource_attr=resource_attr,
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
            searcher=RegularizedEvolution(
                config_space=config_space, metric=metric,
                population_size=population_size, sample_size=sample_size,
                **kwargs),
            **kwargs,
        )


class ConstrainedBayesianOptimization(FIFOScheduler):
    def __init__(
            self, config_space: Dict, metric: str, constraint_attr: str,
            **kwargs):
        search_options = kwargs.get('search_options', dict())
        kwargs['search_options'] = dict(
            search_options, constraint_attr=constraint_attr)
        super(ConstrainedBayesianOptimization, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="bayesopt_constrained",
            **kwargs,
        )


class MSRRandom(MedianStoppingRule):
    def __init__(
            self, config_space: Dict, metric: str, resource_attr: str,
            **kwargs):
        msr_kwargs = dict()
        for k in ('running_average', 'grace_time', 'grace_population',
                  'rank_cutoff'):
            if k in kwargs:
                msr_kwargs[k] = kwargs.pop(k)
        scheduler = RandomSearch(config_space, metric, **kwargs)
        super(MSRRandom, self).__init__(
            scheduler=scheduler,
            resource_attr=resource_attr,
            **msr_kwargs)


# dictionary that allows to also list baselines who don't need a wrapper class such as PBT.
baselines_dict = {
    'Random Search': RandomSearch,
    'Bayesian Optimization': BayesianOptimization,
    'ASHA': ASHA,
    'MOBSTER': MOBSTER,
    'PASHA': PASHA,
    'MOASHA': MOASHA,
    'PBT': PopulationBasedTraining,
    'BORE': BORE,
    'REA': REA,
    'SyncHyperband': SyncHyperband,
    'SyncBOHB': SyncBOHB,
    'SyncMOBSTER': SyncMOBSTER,
    'ConstrainedBayesianOptimization': ConstrainedBayesianOptimization,
}
