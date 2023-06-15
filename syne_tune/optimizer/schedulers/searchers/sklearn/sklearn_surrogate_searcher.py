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
import logging
from typing import Optional, List, Dict, Any

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import (
    EIAcquisitionFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.sklearn_model import (
    SKLearnEstimatorWrapper,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.estimator import (
    SKLearnEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    ScoringFunctionConstructor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults import (
    DEFAULT_NUM_INITIAL_CANDIDATES,
    DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import (
    decode_state,
)
from syne_tune.optimizer.schedulers.searchers.model_based_searcher import (
    BayesianOptimizationSearcher,
)
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges

logger = logging.getLogger(__name__)


class SKLearnSurrogateSearcher(BayesianOptimizationSearcher):
    """SKLearn Surrogate Bayesian optimization for FIFO scheduler

    This searcher must be used with
    :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. It provides
    Bayesian optimization, based on a scikit-learn estimator based surrogate model.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.StochasticSearcher`:

    :param estimator: Instance of
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.estimator.SKLearnEstimator`
        to be used as surrogate model
    :param scoring_class: The scoring function (or acquisition
        function) class and any extra parameters used to instantiate it. If
        ``None``, expected improvement (EI) is used. Note that the acquisition
        function is not locally optimized with this searcher.
    :param num_initial_candidates: Number of candidates sampled for scoring with
        acquisition function.
    :param num_initial_random_choices: Number of randomly chosen candidates before
        surrogate model is used.
    :param allow_duplicates: If ``True``, allow for the same candidate to be
        selected more than once.
    :param restrict_configurations: If given, the searcher only suggests
        configurations from this list. If ``allow_duplicates == False``,
        entries are popped off this list once suggested.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        estimator: SKLearnEstimator,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        scoring_class: Optional[ScoringFunctionConstructor] = None,
        num_initial_candidates: int = DEFAULT_NUM_INITIAL_CANDIDATES,
        num_initial_random_choices: int = DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
        allow_duplicates: bool = False,
        restrict_configurations: Optional[List[Dict[str, Any]]] = None,
        clone_from_state: bool = False,
        **kwargs,
    ):
        super().__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            random_seed_generator=kwargs.get("random_seed_generator"),
            random_seed=kwargs.get("random_seed"),
        )
        self.estimator = SKLearnEstimatorWrapper(estimator)

        if scoring_class is None:
            scoring_class = EIAcquisitionFunction

        if not clone_from_state:
            hp_ranges = make_hyperparameter_ranges(self.config_space)
            self._create_internal(
                hp_ranges=hp_ranges,
                estimator=self.estimator,
                acquisition_class=scoring_class,
                num_initial_candidates=num_initial_candidates,
                num_initial_random_choices=num_initial_random_choices,
                initial_scoring="acq_func",
                skip_local_optimization=True,
                allow_duplicates=allow_duplicates,
                restrict_configurations=restrict_configurations,
                **kwargs,
            )
        else:
            # Internal constructor, bypassing the factory
            # Note: Members which are part of the mutable state, will be
            # overwritten in ``_restore_from_state``
            self._create_internal(**kwargs.copy())

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state["state"], self._hp_ranges_in_state())
        estimator = self.state_transformer.estimator
        # Call internal constructor
        new_searcher = SKLearnSurrogateSearcher(
            **self._new_searcher_kwargs_for_clone(),
            estimator=estimator,
            init_state=init_state,
        )
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
