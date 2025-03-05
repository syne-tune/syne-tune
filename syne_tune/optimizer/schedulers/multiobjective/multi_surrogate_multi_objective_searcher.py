import logging
from functools import partial
from typing import Optional, List, Dict, Any, Union

from syne_tune.optimizer.schedulers.multiobjective.random_scalarization import (
    MultiObjectiveLCBRandomLinearScalarization,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    TrialEvaluations,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator import Estimator
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    ScoringFunctionConstructor,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults import (
    DEFAULT_NUM_INITIAL_CANDIDATES,
    DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
)
from syne_tune.optimizer.schedulers.searchers.legacy_gp_searcher_utils import (
    decode_state,
)
from syne_tune.optimizer.schedulers.searchers.legacy_model_based_searcher import (
    BayesianOptimizationSearcher,
)
from syne_tune.optimizer.schedulers.searchers.searcher_base import extract_random_seed
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges

logger = logging.getLogger(__name__)


class MultiObjectiveMultiSurrogateSearcher(BayesianOptimizationSearcher):
    """Multi Objective Multi Surrogate Searcher for FIFO scheduler

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
        metric: List[str],
        estimators: Dict[str, Estimator],
        mode: Union[List[str], str] = "min",
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
        scoring_class: Optional[ScoringFunctionConstructor] = None,
        num_initial_candidates: int = DEFAULT_NUM_INITIAL_CANDIDATES,
        num_initial_random_choices: int = DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
        allow_duplicates: bool = False,
        restrict_configurations: Optional[List[Dict[str, Any]]] = None,
        clone_from_state: bool = False,
        **kwargs,
    ):
        if mode is None:
            mode = "min"

        super().__init__(
            config_space,
            metric=metric,
            mode=mode,
            points_to_evaluate=points_to_evaluate,
            random_seed_generator=kwargs.get("random_seed_generator"),
            random_seed=kwargs.get("random_seed"),
        )
        if scoring_class is None:
            random_seed, _ = extract_random_seed(**kwargs)
            scoring_class = partial(
                MultiObjectiveLCBRandomLinearScalarization, random_seed=random_seed
            )
        self._set_metric_to_internal_signs()

        if not clone_from_state:
            hp_ranges = make_hyperparameter_ranges(self.config_space)
            self._create_internal(
                hp_ranges=hp_ranges,
                estimator=estimators,
                acquisition_class=scoring_class,
                num_initial_candidates=num_initial_candidates,
                num_initial_random_choices=num_initial_random_choices,
                initial_scoring="acq_func",
                skip_local_optimization={name: True for name in estimators.keys()},
                allow_duplicates=allow_duplicates,
                restrict_configurations=restrict_configurations,
            )
        else:
            # Internal constructor, bypassing the factory
            # Note: Members which are part of the mutable state, will be
            # overwritten in ``_restore_from_state``
            self._create_internal(**kwargs.copy())

    def _set_metric_to_internal_signs(self):
        num_metrics = len(self._metric)
        if isinstance(self._mode, str):
            mode = [self._mode] * num_metrics
        else:
            assert (
                len(self._mode) == num_metrics
            ), f"mode = {self._mode} and metric = {self._metric} must have the same length"
            mode = self._mode
        self._metric_to_internal_signs = [-1 if m == "max" else 1 for m in mode]

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
        # Gather metrics
        metrics = {
            name: result[name] * self._metric_to_internal_signs[pos]
            for pos, name in enumerate(self._metric)
        }

        self.state_transformer.label_trial(
            TrialEvaluations(trial_id=trial_id, metrics=metrics), config=config
        )
        if self.debug_log is not None:
            _trial_id = self._trial_id_string(trial_id, result)
            part = ",".join([f"{name}:{result[name]:.3f}" for name in self._metric])
            msg = f"Update for trial_id {_trial_id}: metrics = ["
            logger.info(msg + part + "]")

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state["state"], self._hp_ranges_in_state())
        estimator = self.state_transformer.estimator
        # Call internal constructor
        new_searcher = MultiObjectiveMultiSurrogateSearcher(
            **self._new_searcher_kwargs_for_clone(),
            estimator=estimator,
            init_state=init_state,
        )
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
