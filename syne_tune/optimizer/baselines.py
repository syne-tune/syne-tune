from typing import Dict, Optional

from syne_tune.optimizer.schedulers import FIFOScheduler, HyperbandScheduler, \
    PopulationBasedTraining
from syne_tune.optimizer.schedulers.multiobjective import MOASHA
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import \
    RegularizedEvolution
from syne_tune.optimizer.schedulers.synchronous import \
    SynchronousGeometricHyperbandScheduler
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations, zero_shot


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
            self, config_space: Dict,
            metric: str,
            resource_attr: str,
            **kwargs,
            ):
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
            self, config_space: Dict,
            metric: str,
            resource_attr: str,
            **kwargs,
            ):
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
            self, config_space: Dict,
            metric: str,
            resource_attr: str,
            **kwargs,
            ):
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


class ZeroShotTransfer(FIFOScheduler):
    def __init__(self, config_space: Dict, transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
                 metric: str, mode: str = 'min', sort_transfer_learning_evaluations: bool = True,
                 use_surrogates: bool = False, random_seed: Optional[int] = None, **kwargs):
        """
        A zero-shot transfer hyperparameter optimization method which jointly selects configurations that minimize the
        average rank obtained on historic metadata (transfer_learning_evaluations).

        Reference: Sequential Model-Free Hyperparameter Tuning.
        Martin Wistuba, Nicolas Schilling, Lars Schmidt-Thieme.
        IEEE International Conference on Data Mining (ICDM) 2015.

        :param config_space: Configuration space for trial evaluation function.
        :param transfer_learning_evaluations: Dictionary from task name to offline evaluations.
        :param metric: Objective name to optimize, must be present in transfer learning evaluations.
        :param mode: Whether to minimize (min) or maximize (max)
        :param sort_transfer_learning_evaluations: Use False if the hyperparameters for each task in
        transfer_learning_evaluations Are already in the same order. If set to True, hyperparameters are sorted.
        :param use_surrogates: If the same configuration is not evaluated on all tasks, set this to true. This will
        generate a set of configurations and will impute their performance using surrogate models.
        :param random_seed: Used for randomly sampling candidates. Only used if use_surrogate is True.
        """
        super(ZeroShotTransfer, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=zero_shot.ZeroShotTransfer(
                config_space=config_space, metric=metric, mode=mode,
                sort_transfer_learning_evaluations=sort_transfer_learning_evaluations, random_seed=random_seed,
                transfer_learning_evaluations=transfer_learning_evaluations, use_surrogates=use_surrogates
            ),
            mode=mode,
            **kwargs,
        )


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
