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
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.sklearn_estimator import (
    SklearnEstimatorWrapper,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.estimator import (
    SklearnEstimator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    ScoringClassAndArgs,
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


class ContributedSurrogateSearcher(BayesianOptimizationSearcher):
    """Contributed Surrogate Bayesian optimization for FIFO scheduler

    This searcher must be used with
    :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. It provides
    Bayesian optimization, based on a Contributed surrogate model.

    Most of the implementation is generic in
    :class:`~syne_tune.optimizer.schedulers.searchers.model_based_searcher.BayesianOptimizationSearcher`.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        estimator: SklearnEstimator,
        scoring_class_and_args: ScoringClassAndArgs = None,
        num_initial_candidates: int = DEFAULT_NUM_INITIAL_CANDIDATES,
        num_initial_random_choices: int = DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
        allow_duplicates: bool = False,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        clone_from_state: bool = False,
        **kwargs,
    ):
        """
        Additional arguments on top of parent class
        :class:`~syne_tune.optimizer.schedulers.searchers.StochasticSearcher`:

        :param estimator: Instance of
            :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.sklearn.estimator.SklearnEstimator`
            to be used as surrogate
        :param scoring_class_and_args: The scoring function (or acquisition function) class
            and any extra parameters used to instantiate it. If None, expected improvement (EI) is used.
            Note that the acquisition function is not locally optimized with this searcher.
        :param num_initial_candidates: Number of candidates sampled for scoring with acquisition function.
        :param num_initial_random_choices: Number of randomly chosen candidates before surrogate model is used.
        :param allow_duplicates: If True, allow for the same candidate to be selected more than once.
        """
        super().__init__(
            config_space,
            metric=metric,
            points_to_evaluate=points_to_evaluate,
            random_seed_generator=kwargs.get("random_seed_generator"),
            random_seed=kwargs.get("random_seed"),
        )
        self.estimator = SklearnEstimatorWrapper(estimator)

        if scoring_class_and_args is None:
            scoring_class_and_args = EIAcquisitionFunction

        if not clone_from_state:
            hp_ranges = make_hyperparameter_ranges(self.config_space)
            self._create_internal(
                hp_ranges=hp_ranges,
                estimator=self.estimator,
                acquisition_class=scoring_class_and_args,
                skip_local_optimization=True,
                initial_scoring="acq_func",
                num_initial_candidates=num_initial_candidates,
                num_initial_random_choices=num_initial_random_choices,
                allow_duplicates=allow_duplicates,
                resource_attr=None,  # TODO This needs to be passed for multi-fidelity
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
        new_searcher = ContributedSurrogateSearcher(
            **self._new_searcher_kwargs_for_clone(),
            estimator=estimator,
            init_state=init_state,
        )
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
