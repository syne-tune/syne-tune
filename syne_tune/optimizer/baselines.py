# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Dict, Optional
import numpy as np
import logging
import copy

from syne_tune.optimizer.schedulers import (
    FIFOScheduler,
    HyperbandScheduler,
    PopulationBasedTraining,
)
from syne_tune.optimizer.schedulers.multiobjective import MOASHA
from syne_tune.optimizer.schedulers.searchers.regularized_evolution import (
    RegularizedEvolution,
)
from syne_tune.optimizer.schedulers.synchronous import (
    SynchronousGeometricHyperbandScheduler,
    GeometricDifferentialEvolutionHyperbandScheduler,
)
from syne_tune.optimizer.schedulers.transfer_learning import (
    TransferLearningTaskEvaluations,
)
from syne_tune.try_import import (
    try_import_blackbox_repository_message,
    try_import_bore_message,
)


class RandomSearch(FIFOScheduler):
    def __init__(self, config_space: dict, metric: str, **kwargs):
        super(RandomSearch, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            **kwargs,
        )


class GridSearch(FIFOScheduler):
    def __init__(self, config_space: dict, metric: str, **kwargs):
        super(GridSearch, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="grid",
            **kwargs,
        )


class BayesianOptimization(FIFOScheduler):
    def __init__(self, config_space: dict, metric: str, **kwargs):
        super(BayesianOptimization, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="bayesopt",
            **kwargs,
        )


def _assert_need_one(kwargs: dict, need_one: Optional[set] = None):
    if need_one is None:
        need_one = {"max_t", "max_resource_attr"}
    assert need_one.intersection(kwargs.keys()), f"Need one of these: {need_one}"


class ASHA(HyperbandScheduler):
    def __init__(self, config_space: dict, metric: str, resource_attr: str, **kwargs):
        """
        One of `max_t`, `max_resource_attr` needs to be in `kwargs`. For
        `type='promotion'`, the latter is more useful, see also
        :class:`HyperbandScheduler`.

        """
        _assert_need_one(kwargs)
        super(ASHA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            resource_attr=resource_attr,
            **kwargs,
        )


class MOBSTER(HyperbandScheduler):
    def __init__(self, config_space: dict, metric: str, resource_attr: str, **kwargs):
        """
        One of `max_t`, `max_resource_attr` needs to be in `kwargs`. For
        `type='promotion'`, the latter is more useful, see also
        :class:`HyperbandScheduler`.

        MOBSTER can be run with different surrogate models. The model is selected
        by `search_options["model"]` in `kwargs`. The default is `"gp_multitask"`
        (jointly dependent multi-task GP model), another useful choice is
        `"gp_independent"` (independent GP models at each rung level, with shared
        ARD kernel).
        """
        _assert_need_one(kwargs)
        super(MOBSTER, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="bayesopt",
            resource_attr=resource_attr,
            **kwargs,
        )


class HyperTune(HyperbandScheduler):
    def __init__(self, config_space: Dict, metric: str, resource_attr: str, **kwargs):
        """
         One of `max_t`, `max_resource_attr` needs to be in `kwargs`. For
         `type='promotion'`, the latter is more useful, see also
         :class:`HyperbandScheduler`.

         Hyper-Tune is a model-based variant of ASHA with more than one bracket.
         It can be seen as extension of MOBSTER and can be used with the
         "gp_independent" or "gp_multitask" model. It has a model-based way
         to sample the bracket for every new trial, as well as an ensemble
        predictive distribution. Our
         implementation is based on:

            Yang Li et al
            Hyper-Tune: Towards Efficient Hyper-parameter Tuning at Scale
            VLDB 2022

        See also :class:`HyperTuneIndependentGPModel`.
        """
        _assert_need_one(kwargs)
        kwargs = copy.deepcopy(kwargs)
        search_options = kwargs.get("search_options", dict())
        k, v, supp = "model", "gp_independent", {"gp_independent", "gp_multitask"}
        model = search_options.get(k, v)
        assert model in supp, (
            f"HyperTune does not support search_options['{k}'] = '{model}'"
            f", must be in {supp}"
        )
        search_options[k] = model
        k = "hypertune_distribution_num_samples"
        num_samples = search_options.get(k, 50)
        search_options[k] = num_samples
        kwargs["search_options"] = search_options
        num_brackets = kwargs.get("brackets", 4)
        kwargs["brackets"] = num_brackets
        super(HyperTune, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="hypertune",
            resource_attr=resource_attr,
            **kwargs,
        )


class PASHA(HyperbandScheduler):
    def __init__(self, config_space: dict, metric: str, resource_attr: str, **kwargs):
        """
        One of `max_t`, `max_resource_attr` needs to be in `kwargs`. The
        latter is more useful, see also :class:`HyperbandScheduler`.
        """
        _assert_need_one(kwargs)
        super(PASHA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            resource_attr=resource_attr,
            type="pasha",
            **kwargs,
        )


class SyncHyperband(SynchronousGeometricHyperbandScheduler):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        resource_attr: str,
        **kwargs,
    ):
        """
        One of `max_resource_level`, `max_resource_attr` needs to be in
        `kwargs`. The latter is more useful, see also
        :class:`HyperbandScheduler`.

        """
        _assert_need_one(kwargs, need_one={"max_resource_level", "max_resource_attr"})
        super(SyncHyperband, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            resource_attr=resource_attr,
            **kwargs,
        )


class SyncBOHB(SynchronousGeometricHyperbandScheduler):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        resource_attr: str,
        **kwargs,
    ):
        """
        One of `max_resource_level`, `max_resource_attr` needs to be in
        `kwargs`. The latter is more useful, see also
        :class:`HyperbandScheduler`.

        """
        _assert_need_one(kwargs, need_one={"max_resource_level", "max_resource_attr"})
        super(SyncBOHB, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="kde",
            resource_attr=resource_attr,
            **kwargs,
        )


class DEHB(GeometricDifferentialEvolutionHyperbandScheduler):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        resource_attr: str,
        **kwargs,
    ):
        """
        One of `max_resource_level`, `max_resource_attr` needs to be in
        `kwargs`. The latter is more useful, see also
        :class:`HyperbandScheduler`.

        """
        _assert_need_one(kwargs, need_one={"max_resource_level", "max_resource_attr"})
        super(DEHB, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="random",
            resource_attr=resource_attr,
            **kwargs,
        )


class SyncMOBSTER(SynchronousGeometricHyperbandScheduler):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        resource_attr: str,
        **kwargs,
    ):
        """
        One of `max_resource_level`, `max_resource_attr` needs to be in
        `kwargs`. The latter is more useful, see also
        :class:`HyperbandScheduler`.

        The default surrogate model is "gp_independent", different to async
        MOBSTER.

        """
        _assert_need_one(kwargs, need_one={"max_resource_level", "max_resource_attr"})
        search_options = kwargs.get("search_options", dict())
        if "model" not in search_options:
            search_options["model"] = "gp_independent"
        kwargs["search_options"] = search_options
        super(SyncMOBSTER, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="bayesopt",
            resource_attr=resource_attr,
            **kwargs,
        )


class BORE(FIFOScheduler):
    def __init__(self, config_space: dict, metric: str, mode: str, **kwargs):
        try:
            from syne_tune.optimizer.schedulers.searchers.bore import Bore
        except ImportError:
            logging.info(try_import_bore_message())
            raise

        super(BORE, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=Bore(config_space=config_space, metric=metric, mode=mode),
            mode=mode,
            **kwargs,
        )


class REA(FIFOScheduler):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        population_size: int = 100,
        sample_size: int = 10,
        **kwargs,
    ):
        super(REA, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=RegularizedEvolution(
                config_space=config_space,
                metric=metric,
                population_size=population_size,
                sample_size=sample_size,
                **kwargs,
            ),
            **kwargs,
        )


class ConstrainedBayesianOptimization(FIFOScheduler):
    def __init__(self, config_space: dict, metric: str, constraint_attr: str, **kwargs):
        search_options = kwargs.get("search_options", dict())
        kwargs["search_options"] = dict(search_options, constraint_attr=constraint_attr)
        super(ConstrainedBayesianOptimization, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher="bayesopt_constrained",
            **kwargs,
        )


class ZeroShotTransfer(FIFOScheduler):
    def __init__(
        self,
        config_space: dict,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        metric: str,
        mode: str = "min",
        sort_transfer_learning_evaluations: bool = True,
        use_surrogates: bool = False,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
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
        try:
            from syne_tune.optimizer.schedulers.transfer_learning import zero_shot
        except ImportError:
            logging.info(try_import_blackbox_repository_message())
            raise

        super(ZeroShotTransfer, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=zero_shot.ZeroShotTransfer(
                config_space=config_space,
                metric=metric,
                mode=mode,
                sort_transfer_learning_evaluations=sort_transfer_learning_evaluations,
                random_seed=random_seed,
                transfer_learning_evaluations=transfer_learning_evaluations,
                use_surrogates=use_surrogates,
            ),
            mode=mode,
            **kwargs,
        )


class ASHACTS(HyperbandScheduler):
    def __init__(
        self,
        config_space: dict,
        metric: str,
        resource_attr: str,
        transfer_learning_evaluations: Dict[str, TransferLearningTaskEvaluations],
        mode: str = "min",
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Runs ASHA where the searcher is done with the transfer-learning method:
        A Quantile-based Approach for Hyperparameter Transfer Learning.
        David Salinas, Huibin Shen, Valerio Perrone. ICML 2020.
        This is the Copula Thompson Sampling approach described in the paper where a surrogate is fitted on the
        transfer learning data to predict mean/variance of configuration performance given a hyperparameter.
        The surrogate is then sampled from and the best configurations are returned as next candidate to evaluate.
        :param config_space:
        :param metric:
        :param resource_attr:
        :param transfer_learning_evaluations:
        :param mode:
        :param random_seed:
        :param kwargs:
        """
        try:
            from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import (
                QuantileBasedSurrogateSearcher,
            )
        except ImportError:
            logging.info(try_import_blackbox_repository_message())
            raise

        super(ASHACTS, self).__init__(
            config_space=config_space,
            searcher=QuantileBasedSurrogateSearcher(
                mode=mode,
                config_space=config_space,
                metric=metric,
                transfer_learning_evaluations=transfer_learning_evaluations,
                random_seed=random_seed
                if random_seed
                else np.random.randint(0, 2**32),
            ),
            mode=mode,
            metric=metric,
            resource_attr=resource_attr,
            **kwargs,
        )


# dictionary that allows to also list baselines who don't need a wrapper class such as PBT.
baselines_dict = {
    "Random Search": RandomSearch,
    "Grid Search": GridSearch,
    "Bayesian Optimization": BayesianOptimization,
    "ASHA": ASHA,
    "MOBSTER": MOBSTER,
    "PASHA": PASHA,
    "MOASHA": MOASHA,
    "PBT": PopulationBasedTraining,
    "BORE": BORE,
    "REA": REA,
    "SyncHyperband": SyncHyperband,
    "SyncBOHB": SyncBOHB,
    "DEHB": DEHB,
    "SyncMOBSTER": SyncMOBSTER,
    "ConstrainedBayesianOptimization": ConstrainedBayesianOptimization,
    "ZeroShotTransfer": ZeroShotTransfer,
    "ASHACTS": ASHACTS,
}
