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
import numpy as np
from typing import Optional, List, Dict, Any
import logging
import copy

from syne_tune.optimizer.schedulers.searchers import ModelBasedSearcher
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import (
    gp_fifo_searcher_factory,
    gp_fifo_searcher_defaults,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import (
    decode_state,
)
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Configuration,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    ModelStateTransformer,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_skipopt import (
    AlwaysSkipPredicate,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    ScoringFunction,
    SurrogateOutputModel,
    AcquisitionClassAndArgs,
    unwrap_acquisition_class_and_kwargs,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm import (
    BayesianOptimizationAlgorithm,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import (
    IndependentThompsonSampling,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common import (
    RandomStatefulCandidateGenerator,
    RandomFromSetCandidateGenerator,
    CandidateGenerator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.duplicate_detector import (
    DuplicateDetectorIdentical,
)

logger = logging.getLogger(__name__)


def create_initial_candidates_scorer(
    initial_scoring: str,
    model: SurrogateOutputModel,
    acquisition_class: AcquisitionClassAndArgs,
    random_state: np.random.RandomState,
    active_output: str = INTERNAL_METRIC_NAME,
) -> ScoringFunction:
    if initial_scoring == "thompson_indep":
        if isinstance(model, dict):
            assert active_output in model
            model = model[active_output]
        return IndependentThompsonSampling(model, random_state=random_state)
    else:
        acquisition_class, acquisition_kwargs = unwrap_acquisition_class_and_kwargs(
            acquisition_class
        )
        return acquisition_class(
            model, active_metric=active_output, **acquisition_kwargs
        )


class GPFIFOSearcher(ModelBasedSearcher):
    """Gaussian process Bayesian optimization for FIFO scheduler

    This searcher must be used with
    :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. It provides
    Bayesian optimization, based on a Gaussian process surrogate model.

    It is *not* recommended creating :class:`GPFIFOSearcher` searcher objects
    directly, but rather to create
    :class:`~syne_tune.optimizer.schedulers.FIFOScheduler` objects with
    ``searcher="bayesopt"``, and passing arguments here in ``search_options``.
    This will use the appropriate functions from
    :mod:``syne_tune.optimizer.schedulers.searchers.gp_searcher_factory`` to
    create components in a consistent way.

    Note: If metric values are to be maximized (``mode-"max"`` in scheduler),
    the searcher uses ``map_reward`` to map metric values to internal
    criterion values, and *minimizes* the latter. The default choice is
    to multiply values by -1.

    Pending configurations (for which evaluation tasks are currently running)
    are dealt with by fantasizing (i.e., target values are drawn from the
    current posterior, and acquisition functions are averaged over this
    sample, see ``num_fantasy_samples``).

    The GP surrogate model uses a Matern 5/2 covariance function with automatic
    relevance determination (ARD) of input attributes, and a constant mean
    function. The acquisition function is expected improvement (EI). All
    hyperparameters of the surrogate model are estimated by empirical Bayes
    (maximizing the marginal likelihood). In general, this hyperparameter
    fitting is the most expensive part of a :meth:`get_config` call.

    The following happens in :meth:`get_config`:

    * For the first ``num_init_random`` calls, a config is drawn at random
      (after ``points_to_evaluate``, which are included in the ``num_init_random``
      initial ones). Afterwards, Bayesian optimization is used, unless there
      are no finished evaluations yet (a surrogate model cannot be fix on no
      data).
    * For BO, model hyperparameter are refit first. This step can be skipped
      (see ``opt_skip_*`` parameters).
    * Next, ``num_init_candidates`` configs are sampled at random (as in random
      search), and ranked by a scoring function (``initial_scoring``).
    * BFGS local optimization is then run starting from the top scoring config,
      where EI is minimized (this is skipped if
      ``skip_local_optimization == True``).

    Note that the full logic of construction based on arguments is given in
    :mod:``syne_tune.optimizer.schedulers.searchers.gp_searcher_factory``. In
    particular, see
    :func:`~syne_tune.optimizer.schedulers.searchers.gp_searcher_factory.gp_fifo_searcher_defaults`
    for default values.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.StochasticSearcher`:

    :param clone_from_state: Internal argument, do not use
    :type clone_from_state: bool
    :param resource_attr: Name of resource attribute in reports. This is
        optional here, but required for multi-fidelity searchers.
        If ``resource_attr`` and ``cost_attr`` are given, cost values are read from
        each report and stored in the state. This allows cost models to be fit
        on more data.
    :type resource_attr: str, optional
    :param cost_attr: Name of cost attribute in data obtained from reporter
        (e.g., elapsed training time). Needed only by cost-aware searchers.
        Depending on whether ``resource_attr`` is given, cost values are read
        from each report or only at the end.
    :type cost_attr: str, optional
    :param num_init_random: Number of initial :meth:`get_config` calls for which
        randomly sampled configs are returned. Afterwards, the model-based
        searcher is used. Defaults to
        :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS`
    :type num_init_random: int, optional
    :param num_init_candidates: Number of initial candidates sampled at
        random in order to seed the model-based search in ``get_config``.
        Defaults to :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_NUM_INITIAL_CANDIDATES`
    :type num_init_candidates: int, optional
    :param num_fantasy_samples: Number of samples drawn for fantasizing
        (latent target values for pending evaluations), defaults to 20
    :type num_fantasy_samples: int, optional
    :param no_fantasizing: If True, fantasizing is not done and pending
        evaluations are ignored. This may lead to loss of diversity in
        decisions. Defaults to ``False``
    :type no_fantasizing: bool, optional
    :param input_warping: If ``True``, we use a warping transform, so the kernel
        function becomes :math:`k(w(x), w(x'))`, where :math:`w(x)` is a warping
        transform parameterized by two non-negative numbers per component, which
        are learned as hyperparameters. See also
        :class:`~syne_tune.optimizer.schedulers.searcher.bayesopt.gpautograd.warping.Warping`.
        Coordinates which belong to categorical hyperparameters, are not warped.
        Defaults to ``False``.
    :type input_warping: bool, optional
    :param boxcox_transform: If ``True``, target values are transformed before
        being fitted with a Gaussian marginal likelihood. This is using the Box-Cox
        transform with a parameter :math:`\lambda`, which is learned alongside
        other parameters of the surrogate model. The transform is :math:`\log y`
        for :math:`\lambda = 0`, and :math:`y - 1` for :math:`\lambda = 1`. This
        option requires the targets to be positive. Defaults to ``False``.
    :type boxcox_transform: bool, optional
    :param initial_scoring: Scoring function to rank initial candidates
        (local optimization of EI is started from top scorer):

        * "thompson_indep": Independent Thompson sampling; randomized score,
          which can increase exploration
        * "acq_func": score is the same (EI) acquisition function which is
          used for local optimization afterwards

        Defaults to
        :const:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_INITIAL_SCORING`
    :type initial_scoring: str, optional
    :param skip_local_optimization: If ``True``, the local gradient-based
        optimization of the acquisition function is skipped, and the
        top-ranked initial candidate (after initial scoring) is returned
        instead. In this case, ``initial_scoring="acq_func"`` makes most
        sense, otherwise the acquisition function will not be used.
        Defaults to False
    :type skip_local_optimization: bool, optional
    :param opt_nstarts: Parameter for surrogate model fitting. Number of
        random restarts. Defaults to 2
    :type opt_nstarts: int, optional
    :param opt_maxiter: Parameter for surrogate model fitting. Maximum
        number of iterations per restart. Defaults to 50
    :type opt_maxiter: int, optional
    :param opt_warmstart: Parameter for surrogate model fitting. If ``True``,
        each fitting is started from the previous optimum. Not recommended
        in general. Defaults to ``False``
    :type opt_warmstart: bool, optional
    :param opt_verbose: Parameter for surrogate model fitting. If ``True``,
        lots of output. Defaults to ``False``
    :type opt_verbose: bool, optional
    :param opt_skip_init_length: Parameter for surrogate model fitting,
        skip predicate. Fitting is never skipped as long as number of
        observations below this threshold. Defaults to 150
    :type opt_skip_init_length: int, optional
    :param opt_skip_period: Parameter for surrogate model fitting, skip
        predicate. If ``>1``, and number of observations above
        ``opt_skip_init_length``, fitting is done only K-th call, and skipped
        otherwise. Defaults to 1 (no skipping)
    :type opt_skip_period: int, optional
    :param allow_duplicates: If ``True``, :meth:`get_config` may return the same
        configuration more than once. Defaults to ``False``
    :type allow_duplicates: bool, optional
    :param restrict_configurations: If given, the searcher only suggests
        configurations from this list. This needs
        ``skip_local_optimization == True``. If ``allow_duplicates == False``,
         entries are popped off this list once suggested.
    :type restrict_configurations: List[dict], optional
    :param map_reward: In the scheduler, the metric may be minimized or
        maximized, but internally, Bayesian optimization is minimizing
        the criterion. ``map_reward`` converts from metric to internal
        criterion:

        * "minus_x": ``criterion = -metric``
        * "<a>_minus_x": ``criterion = <a> - metric``. For example "1_minus_x"
          maps accuracy to zero-one error

        From a technical standpoint, it does not matter what is chosen here,
        because criterion is only used internally. Also note that criterion
        data is always normalized to mean 0, variance 1 before fitted with a
        Gaussian process. Defaults to "1_minus_x"
    :type map_reward: str or :class:`MapReward`, optional
    :param transfer_learning_task_attr: Used to support transfer HPO, where
        the state contains observed data from several tasks, one of which
        is the active one. To this end, ``config_space`` must contain a
        categorical parameter of name ``transfer_learning_task_attr``, whose
        range are all task IDs. Also, ``transfer_learning_active_task`` must
        denote the active task, and ``transfer_learning_active_config_space``
        is used as ``active_config_space`` argument in
        :class:`~syne_tune.optimizer.schedulers.searchers.utils.HyperparameterRanges`.
        This allows us to use a narrower search space for the active task than
        for the union of all tasks (``config_space`` must be that), which is
        needed if some configurations of non-active tasks lie outside of the
        ranges in ``active_config_space``. One of the implications is that
        :meth:`filter_observed_data` is selecting configs of the active task,
        so that incumbents or exclusion lists are restricted to data from the
        active task.
    :type transfer_learning_task_attr: str, optional
    :param transfer_learning_active_task: See ``transfer_learning_task_attr``.
    :type transfer_learning_active_task: str, optional
    :param transfer_learning_active_config_space:
        See ``transfer_learning_task_attr``. If not given, ``config_space`` is the
        search space for the active task as well. This active config space need
        not contain the ``transfer_learning_task_attr`` parameter. In fact, this
        parameter is set to a categorical with ``transfer_learning_active_task``
        as single value, so that new configs are chosen for the active task
        only.
    :type transfer_learning_active_config_space: Dict[str, Any], optional
    :param transfer_learning_model: See ``transfer_learning_task_attr``.
        Specifies the surrogate model to be used for transfer learning:

        * "matern52_product": Kernel is product of Matern 5/2 (not ARD) on
          ``transfer_learning_task_attr`` and Matern 5/2 (ARD) on the rest.
          Assumes that data from same task are more closely related than
          data from different tasks
        * "matern52_same": Kernel is Matern 5/2 (ARD) on the rest of the
          variables, ``transfer_learning_task_attr`` is ignored. Assumes
          that data from all tasks can be merged together

        Defaults to "matern52_product"
    :type transfer_learning_model: str, optional
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
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
        if not clone_from_state:
            kwargs["config_space"] = config_space
            kwargs["metric"] = metric
            kwargs_int = self._create_kwargs_int(kwargs)
        else:
            # Internal constructor, bypassing the factory
            # Note: Members which are part of the mutable state, will be
            # overwritten in ``_restore_from_state``
            kwargs_int = kwargs.copy()
        self._call_create_internal(kwargs_int)

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *gp_fifo_searcher_defaults(kwargs), dict_name="search_options"
        )
        kwargs_int = gp_fifo_searcher_factory(**_kwargs)
        # Extra arguments not parsed in factory
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def _call_create_internal(self, kwargs_int):
        """
        Part of constructor which can be different in subclasses
        """
        self._create_internal(**kwargs_int)

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.scheduler_searcher import (
            TrialSchedulerWithSearcher,
        )

        assert isinstance(
            scheduler, TrialSchedulerWithSearcher
        ), "This searcher requires TrialSchedulerWithSearcher scheduler"
        super().configure_scheduler(scheduler)
        # Allow model factory to depend on ``scheduler`` as well
        model_factory = self.state_transformer.model_factory
        if isinstance(model_factory, dict):
            model_factories = list(model_factory.values())
        else:
            model_factories = [model_factory]
        for model_factory in model_factories:
            model_factory.configure_scheduler(scheduler)

    def register_pending(
        self, trial_id: str, config: Optional[Dict[str, Any]] = None, milestone=None
    ):
        """
        Registers trial as pending. This means the corresponding evaluation
        task is running. Once it finishes, update is called for this trial.

        """
        # It is OK for the candidate already to be registered as pending, in
        # which case we do nothing
        state = self.state_transformer.state
        if not state.is_pending(trial_id):
            assert not state.is_labeled(trial_id), (
                f"Trial trial_id = {trial_id} is already labeled, so cannot "
                + "be pending"
            )
            self.state_transformer.append_trial(trial_id, config=config)

    def _fix_resource_attribute(self, **kwargs):
        pass

    def _postprocess_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return config

    def _create_random_generator(self) -> CandidateGenerator:
        if self._restrict_configurations is None:
            return RandomStatefulCandidateGenerator(
                hp_ranges=self._hp_ranges_for_prediction(),
                random_state=self.random_state,
            )
        else:
            hp_ranges = self._hp_ranges_for_prediction()
            if hp_ranges.is_attribute_fixed():
                ext_config = {hp_ranges.name_last_pos: hp_ranges.value_for_last_pos}
            else:
                ext_config = None
            return RandomFromSetCandidateGenerator(
                base_set=self._restrict_configurations,
                random_state=self.random_state,
                ext_config=ext_config,
            )

    def _update_restrict_configurations(
        self,
        new_configs: List[Dict[str, Any]],
        random_generator: RandomFromSetCandidateGenerator,
    ):
        # ``random_generator`` maintains all positions returned during the
        # search. This is used to restrict the search of ``new_configs``
        new_ms = set(
            self.hp_ranges.config_to_match_string(config) for config in new_configs
        )
        num_new = len(new_configs)
        assert num_new >= 1
        remove_pos = []
        for pos in random_generator.pos_returned:
            config_ms = self.hp_ranges.config_to_match_string(
                self._restrict_configurations[pos]
            )
            if config_ms in new_ms:
                remove_pos.append(pos)
                if len(remove_pos) == num_new:
                    break
        if len(remove_pos) == 1:
            self._restrict_configurations.pop(remove_pos[0])
        else:
            remove_pos = set(remove_pos)
            self._restrict_configurations = [
                config
                for pos, config in enumerate(self._restrict_configurations)
                if pos not in remove_pos
            ]

    def _get_config_modelbased(
        self, exclusion_candidates, **kwargs
    ) -> Optional[Configuration]:
        # Obtain current :class:`SurrogateModel` from state transformer. Based on
        # this, the BO algorithm components can be constructed
        if self.do_profile:
            self.profiler.push_prefix("getconfig")
            self.profiler.start("all")
            self.profiler.start("gpmodel")
        # Note: Asking for the model triggers the posterior computation
        model = self.state_transformer.model()
        if self.do_profile:
            self.profiler.stop("gpmodel")
        # Select and fix target resource attribute (relevant in subclasses)
        self._fix_resource_attribute(**kwargs)
        # Create BO algorithm
        random_generator = self._create_random_generator()
        initial_candidates_scorer = create_initial_candidates_scorer(
            initial_scoring=self.initial_scoring,
            model=model,
            acquisition_class=self.acquisition_class,
            random_state=self.random_state,
        )
        local_optimizer = self.local_minimizer_class(
            hp_ranges=self._hp_ranges_for_prediction(),
            model=model,
            acquisition_class=self.acquisition_class,
            active_metric=INTERNAL_METRIC_NAME,
        )
        bo_algorithm = BayesianOptimizationAlgorithm(
            initial_candidates_generator=random_generator,
            initial_candidates_scorer=initial_candidates_scorer,
            num_initial_candidates=self.num_initial_candidates,
            local_optimizer=local_optimizer,
            pending_candidate_state_transformer=None,
            exclusion_candidates=exclusion_candidates,
            num_requested_candidates=1,
            greedy_batch_selection=False,
            duplicate_detector=DuplicateDetectorIdentical(),
            profiler=self.profiler,
            sample_unique_candidates=False,
            debug_log=self.debug_log,
        )
        # Next candidate decision
        _config = bo_algorithm.next_candidates()
        if len(_config) > 0:
            config = self._postprocess_config(_config[0])
            if self._restrict_configurations is not None:
                # Remove ``config`` from ``_restrict_configurations``
                self._update_restrict_configurations([config], random_generator)
        else:
            config = None
        if self.do_profile:
            self.profiler.stop("all")
            self.profiler.pop_prefix()  # getconfig
        return config

    def get_batch_configs(
        self,
        batch_size: int,
        num_init_candidates_for_batch: Optional[int] = None,
        **kwargs,
    ) -> List[Configuration]:
        """
        Asks for a batch of ``batch_size`` configurations to be suggested. This
        is roughly equivalent to calling ``get_config`` ``batch_size`` times,
        marking the suggested configs as pending in the state (but the state
        is not modified here). This means the batch is chosen sequentially,
        at about the cost of calling ``get_config`` ``batch_size`` times.

        If ``num_init_candidates_for_batch`` is given, it is used instead
        of ``num_init_candidates`` for the selection of all but the first
        config in the batch. In order to speed up batch selection, choose
        ``num_init_candidates_for_batch`` smaller than
        ``num_init_candidates``.

        If less than ``batch_size`` configs are returned, the search space
        has been exhausted.

        Note: Batch selection does not support ``debug_log`` right now: make sure
        to switch this off when creating scheduler and searcher.
        """
        assert round(batch_size) == batch_size and batch_size >= 1
        configs = []
        if batch_size == 1:
            config = self.get_config(**kwargs)
            if config is not None:
                configs.append(config)
        else:
            # :class:`DebugLogWriter` does not support batch selection right now,
            # must be switched off
            assert self.debug_log is None, (
                "``get_batch_configs`` does not support debug_log right now. "
                + "Please set ``debug_log=False`` in search_options argument "
                + "of scheduler, or create your searcher with ``debug_log=False``"
            )
            exclusion_candidates = self._get_exclusion_candidates(
                skip_observed=self._allow_duplicates
            )
            pick_random = True
            while pick_random and len(configs) < batch_size:
                config, pick_random = self._get_config_not_modelbased(
                    exclusion_candidates
                )
                if pick_random:
                    if config is not None:
                        configs.append(config)
                        # Even if ``allow_duplicates == True``, we don't want to have
                        # duplicates in the same batch
                        exclusion_candidates.add(config)
                    else:
                        break  # Space exhausted
            if not pick_random:
                # Model-based decision for remaining ones
                num_requested_candidates = batch_size - len(configs)
                model = self.state_transformer.model()
                # Select and fix target resource attribute (relevant in subclasses)
                self._fix_resource_attribute(**kwargs)
                # Create BO algorithm
                random_generator = self._create_random_generator()
                initial_candidates_scorer = create_initial_candidates_scorer(
                    initial_scoring=self.initial_scoring,
                    model=model,
                    acquisition_class=self.acquisition_class,
                    random_state=self.random_state,
                )
                local_optimizer = self.local_minimizer_class(
                    hp_ranges=self._hp_ranges_for_prediction(),
                    model=model,
                    acquisition_class=self.acquisition_class,
                    active_metric=INTERNAL_METRIC_NAME,
                )
                pending_candidate_state_transformer = None
                if num_requested_candidates > 1:
                    # Internally, if num_requested_candidates > 1, the candidates are
                    # selected greedily. This needs model updates after each greedy
                    # selection, because of one more pending evaluation.
                    # We need a copy of the state here, since
                    # ``pending_candidate_state_transformer`` modifies the state (it
                    # appends pending trials)
                    temporary_state = copy.deepcopy(self.state_transformer.state)
                    pending_candidate_state_transformer = ModelStateTransformer(
                        model_factory=self.state_transformer.model_factory,
                        init_state=temporary_state,
                        skip_optimization=AlwaysSkipPredicate(),
                    )
                bo_algorithm = BayesianOptimizationAlgorithm(
                    initial_candidates_generator=random_generator,
                    initial_candidates_scorer=initial_candidates_scorer,
                    num_initial_candidates=self.num_initial_candidates,
                    num_initial_candidates_for_batch=num_init_candidates_for_batch,
                    local_optimizer=local_optimizer,
                    pending_candidate_state_transformer=pending_candidate_state_transformer,
                    exclusion_candidates=exclusion_candidates,
                    num_requested_candidates=num_requested_candidates,
                    greedy_batch_selection=True,
                    duplicate_detector=DuplicateDetectorIdentical(),
                    sample_unique_candidates=False,
                    debug_log=self.debug_log,
                )
                # Next candidate decision
                _configs = [
                    self._postprocess_config(config)
                    for config in bo_algorithm.next_candidates()
                ]
                configs.extend(_configs)
                if self._restrict_configurations is not None:
                    self._update_restrict_configurations(_configs, random_generator)
        return configs

    def evaluation_failed(self, trial_id: str):
        # Remove pending evaluation
        self.state_transformer.drop_pending_evaluation(trial_id)
        # Mark config as failed (which means it will be blacklisted in
        # future get_config calls)
        self.state_transformer.mark_trial_failed(trial_id)

    def _new_searcher_kwargs_for_clone(self) -> Dict[str, Any]:
        """
        Helper method for ``clone_from_state``. Args need to be extended
        by ``model_factory``, ``init_state``, ``skip_optimization``, and others
        args becoming relevant in subclasses only.

        :return: kwargs for creating new searcher object in ``clone_from_state``
        """
        return dict(
            config_space=self.config_space,
            metric=self._metric,
            clone_from_state=True,
            hp_ranges=self.hp_ranges,
            acquisition_class=self.acquisition_class,
            map_reward=self.map_reward,
            local_minimizer_class=self.local_minimizer_class,
            num_initial_candidates=self.num_initial_candidates,
            num_initial_random_choices=self.num_initial_random_choices,
            initial_scoring=self.initial_scoring,
            skip_local_optimization=self.skip_local_optimization,
            cost_attr=self._cost_attr,
            resource_attr=self._resource_attr,
            filter_observed_data=self._filter_observed_data,
            allow_duplicates=self._allow_duplicates,
            restrict_configurations=self._restrict_configurations,
        )

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state["state"], self._hp_ranges_in_state())
        skip_optimization = state["skip_optimization"]
        model_factory = self.state_transformer.model_factory
        # Call internal constructor
        new_searcher = GPFIFOSearcher(
            **self._new_searcher_kwargs_for_clone(),
            model_factory=model_factory,
            init_state=init_state,
            skip_optimization=skip_optimization,
        )
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
