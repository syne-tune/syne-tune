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
from typing import Type, Optional, List
import logging
import copy
import time

from syne_tune.optimizer.schedulers.searchers.searcher import (
    SearcherWithRandomSeed,
    RandomSearcher,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import (
    gp_fifo_searcher_factory,
    gp_fifo_searcher_defaults,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import (
    DEFAULT_INITIAL_SCORING,
    SUPPORTED_INITIAL_SCORING,
    MapReward,
    encode_state,
    decode_state,
)
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    TrialEvaluations,
    MetricValues,
    dictionarize_objective,
    INTERNAL_METRIC_NAME,
    INTERNAL_COST_NAME,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Configuration,
    ConfigurationFilter,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    TransformerModelFactory,
    ModelStateTransformer,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_skipopt import (
    SkipOptimizationPredicate,
    AlwaysSkipPredicate,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    LocalOptimizer,
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
    NoOptimization,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common import (
    RandomStatefulCandidateGenerator,
    ExclusionList,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults import (
    DEFAULT_LOCAL_OPTIMIZER_CLASS,
    DEFAULT_NUM_INITIAL_CANDIDATES,
    DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.duplicate_detector import (
    DuplicateDetectorIdentical,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler

logger = logging.getLogger(__name__)


GET_CONFIG_RANDOM_RETRIES = 50


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


def check_initial_candidates_scorer(initial_scoring: Optional[str]) -> str:
    if initial_scoring is None:
        return DEFAULT_INITIAL_SCORING
    else:
        assert (
            initial_scoring in SUPPORTED_INITIAL_SCORING
        ), "initial_scoring = '{}' is not supported".format(initial_scoring)
        return initial_scoring


class ModelBasedSearcher(SearcherWithRandomSeed):
    """Common code for surrogate model based searchers"""

    def _create_internal(
        self,
        hp_ranges: HyperparameterRanges,
        model_factory: TransformerModelFactory,
        acquisition_class: AcquisitionClassAndArgs,
        map_reward: Optional[MapReward] = None,
        init_state: TuningJobState = None,
        local_minimizer_class: Type[LocalOptimizer] = None,
        skip_optimization: SkipOptimizationPredicate = None,
        num_initial_candidates: int = DEFAULT_NUM_INITIAL_CANDIDATES,
        num_initial_random_choices: int = DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
        initial_scoring: Optional[str] = None,
        skip_local_optimization: bool = False,
        cost_attr: Optional[str] = None,
        resource_attr: Optional[str] = None,
        filter_observed_data: Optional[ConfigurationFilter] = None,
    ):
        self.hp_ranges = hp_ranges
        self.num_initial_candidates = num_initial_candidates
        self.num_initial_random_choices = num_initial_random_choices
        self.map_reward = map_reward
        if skip_local_optimization:
            self.local_minimizer_class = NoOptimization
        elif local_minimizer_class is None:
            self.local_minimizer_class = DEFAULT_LOCAL_OPTIMIZER_CLASS
        else:
            self.local_minimizer_class = local_minimizer_class
        self.acquisition_class = acquisition_class
        self._debug_log = model_factory.debug_log
        self.initial_scoring = check_initial_candidates_scorer(initial_scoring)
        self.skip_local_optimization = skip_local_optimization
        # Create state transformer
        # Initial state is empty (note that the state is mutable)
        if init_state is None:
            init_state = TuningJobState.empty_state(self._hp_ranges_in_state())
        self.state_transformer = ModelStateTransformer(
            model_factory=model_factory,
            init_state=init_state,
            skip_optimization=skip_optimization,
        )
        self.random_generator = RandomStatefulCandidateGenerator(
            self._hp_ranges_for_prediction(), random_state=self.random_state
        )
        self.set_profiler(model_factory.profiler)
        self._cost_attr = cost_attr
        self._resource_attr = resource_attr
        self._filter_observed_data = filter_observed_data
        self._random_searcher = None
        # Tracks the cumulative time spent in ``get_config`` calls
        self.cumulative_get_config_time = 0
        if model_factory.debug_log is not None:
            deb_msg = "[ModelBasedSearcher.__init__]\n"
            deb_msg += "- acquisition_class = {}\n".format(acquisition_class)
            deb_msg += "- local_minimizer_class = {}\n".format(local_minimizer_class)
            deb_msg += "- num_initial_candidates = {}\n".format(num_initial_candidates)
            deb_msg += "- num_initial_random_choices = {}\n".format(
                num_initial_random_choices
            )
            deb_msg += "- initial_scoring = {}\n".format(self.initial_scoring)
            logger.info(deb_msg)

    def _copy_kwargs_to_kwargs_int(self, kwargs_int: dict, kwargs: dict):
        """Copies extra arguments not dealt with by ``gp_fifo_searcher_factory``

        :param kwargs_int: Output of factory, to be passed to ``searcher_factory``
        :param kwargs: Input arguments
        """
        # Extra arguments not parsed in factory
        for k in ("init_state", "local_minimizer_class", "cost_attr", "resource_attr"):
            kwargs_int[k] = kwargs.get(k)

    def _hp_ranges_in_state(self):
        """
        :return: ``HyperparameterRanges`` to be used in ``self.state_transformer.state``
        """
        return self.hp_ranges

    def _hp_ranges_for_prediction(self):
        """
        :return: ``HyperparameterRanges`` to be used in predictions and acquisition
            functions
        """
        return self._hp_ranges_in_state()

    def _metric_val_update(self, crit_val: float, result: dict) -> MetricValues:
        return crit_val

    def on_trial_result(self, trial_id: str, config: dict, result: dict, update: bool):
        # If both ``cost_attr`` and ``resource_attr`` are given, cost data (if
        # given) is written out from every ``result``, independent of ``update``
        cattr = self._cost_attr
        rattr = self._resource_attr
        if (
            cattr is not None
            and rattr is not None
            and cattr in result
            and rattr in result
        ):
            cost_val = float(result[cattr])
            resource = str(result[rattr])
            metrics = {INTERNAL_COST_NAME: {resource: cost_val}}
            self.state_transformer.label_trial(
                TrialEvaluations(trial_id=trial_id, metrics=metrics), config=config
            )
        if update:
            self._update(trial_id, config, result)

    def _trial_id_string(self, trial_id: str, result: dict):
        """
        For multi-fidelity, we also want to output the resource level
        """
        return trial_id

    def _update(self, trial_id: str, config: dict, result: dict):
        metric_val = result[self._metric]
        # Transform to criterion to be minimized
        if self.map_reward is not None:
            crit_val = self.map_reward(metric_val)
        else:
            crit_val = metric_val
        metrics = dictionarize_objective(self._metric_val_update(crit_val, result))
        # Cost value only dealt with here if ``resource_attr`` not given
        attr = self._cost_attr
        cost_val = None
        if attr is not None and attr in result:
            cost_val = float(result[attr])
            if self._resource_attr is None:
                metrics[INTERNAL_COST_NAME] = cost_val
        self.state_transformer.label_trial(
            TrialEvaluations(trial_id=trial_id, metrics=metrics), config=config
        )
        if self.debug_log is not None:
            _trial_id = self._trial_id_string(trial_id, result)
            msg = f"Update for trial_id {_trial_id}: metric = {metric_val:.3f}"
            if self.map_reward is not None:
                msg += f", crit_val = {crit_val:.3f}"
            if cost_val is not None:
                msg += f", cost_val = {cost_val:.2f}"
            logger.info(msg)

    def _get_config_modelbased(
        self, exclusion_candidates: ExclusionList, **kwargs
    ) -> Optional[Configuration]:
        """
        Implements ``get_config`` part if the surrogate model is used, instead
        of initial choices from ``points_to_evaluate`` or initial random
        choices.

        :param exclusion_candidates: Configs to be avoided
        :param kwargs: Extra arguments
        :return: Suggested configuration, or None if configuration space is
            exhausted
        """
        raise NotImplementedError

    def _get_exclusion_candidates(self, **kwargs) -> ExclusionList:
        return ExclusionList(
            self.state_transformer.state,
            filter_observed_data=self._filter_observed_data,
        )

    def _should_pick_random_config(self, exclusion_candidates: ExclusionList) -> bool:
        """
        :param exclusion_candidates: Configs to be avoided
        :return: Should config be drawn at random in ``get_config``
        """
        if len(exclusion_candidates) < self.num_initial_random_choices:
            return True
        # Determine whether there is any observed data after filtering
        state = self.state_transformer.state
        if not state.trials_evaluations:
            return True
        if self._filter_observed_data is None:
            return False
        for ev in state.trials_evaluations:
            config = state.config_for_trial[ev.trial_id]
            if self._filter_observed_data(config):
                return False
        return True

    def _get_config_not_modelbased(
        self, exclusion_candidates: ExclusionList
    ) -> (Optional[Configuration], bool):
        """
        Does job of ``get_config``, as long as the decision does not involve
        model-based search. If False is returned, model-based search must be
        called.

        :param exclusion_candidates: Configs to be avoided
        :return: ``(config, use_get_config_modelbased)``
        """
        self._assign_random_searcher()
        config = self._next_initial_config()  # Ask for initial config
        if config is None:
            pick_random = self._should_pick_random_config(exclusion_candidates)
        else:
            pick_random = True  # Initial configs count as random here
        if pick_random and config is None:
            if self.do_profile:
                self.profiler.start("random")
            for _ in range(GET_CONFIG_RANDOM_RETRIES):
                _config = self._random_searcher.get_config()
                if _config is None:
                    # If ``RandomSearcher`` returns no config at all, the
                    # search space is exhausted
                    break
                if not exclusion_candidates.contains(_config):
                    config = _config
                    break
            if self.do_profile:
                self.profiler.stop("random")
        return config, pick_random

    def get_config(self, **kwargs) -> Optional[dict]:
        """
        Runs Bayesian optimization in order to suggest the next config to evaluate.

        :return: Next config to evaluate at
        """
        start_time = time.time()
        state = self.state_transformer.state
        if self.do_profile:
            # Start new profiler block
            skip_optimization = self.state_transformer.skip_optimization
            if isinstance(skip_optimization, dict):
                skip_optimization = skip_optimization[INTERNAL_METRIC_NAME]
            meta = {
                "fit_hyperparams": not skip_optimization(state),
                "num_observed": state.num_observed_cases(),
                "num_pending": len(state.pending_evaluations),
            }
            self.profiler.begin_block(meta)
            self.profiler.start("all")
            # Initial configs come from ``points_to_evaluate`` or are drawn at random
        exclusion_candidates = self._get_exclusion_candidates(**kwargs)
        config, pick_random = self._get_config_not_modelbased(exclusion_candidates)
        if self.debug_log is not None:
            trial_id = kwargs.get("trial_id")
            self.debug_log.start_get_config(
                "random" if pick_random else "BO", trial_id=trial_id
            )
        if not pick_random:
            # Model-based decision
            if not exclusion_candidates.config_space_exhausted():
                config = self._get_config_modelbased(exclusion_candidates, **kwargs)

        if config is not None:
            if self.debug_log is not None:
                self.debug_log.set_final_config(config)
                # All get_config debug log info is only written here
                self.debug_log.write_block()
        else:
            msg = (
                "Failed to sample a configuration not already chosen "
                + f"before. Exclusion list has size {len(exclusion_candidates)}."
            )
            cs_size = exclusion_candidates.configspace_size
            if cs_size is not None:
                msg += f" Configuration space has size {cs_size}."
            logger.warning(msg)
        if self.do_profile:
            self.profiler.stop("all")
            self.profiler.clear()
        self.cumulative_get_config_time += time.time() - start_time

        return config

    def dataset_size(self):
        return self.state_transformer.state.num_observed_cases()

    def model_parameters(self):
        return self.state_transformer.get_params()

    def set_params(self, param_dict):
        self.state_transformer.set_params(param_dict)

    def get_state(self) -> dict:
        """
        The mutable state consists of the GP model parameters, the
        TuningJobState, and the skip_optimization predicate (which can have a
        mutable state).
        We assume that skip_optimization can be pickled.

        """
        state = dict(
            super().get_state(),
            model_params=self.model_parameters(),
            state=encode_state(self.state_transformer.state),
            skip_optimization=self.state_transformer.skip_optimization,
        )
        if self._random_searcher is not None:
            state["random_searcher_state"] = self._random_searcher.get_state()
        return state

    def _restore_from_state(self, state: dict):
        super()._restore_from_state(state)
        self.state_transformer.set_params(state["model_params"])
        self.random_generator.random_state = self.random_state
        if "random_searcher_state" in state:
            # Restore self._random_searcher as well
            # Note: It is important to call ``_assign_random_searcher`` with a
            # random seed. Otherwise, one is drawn from ``random_state``, which
            # modifies that state. The seed passed does not matter, since
            # ``_random_searcher.random_state`` will be restored anyway
            self._assign_random_searcher(random_seed=0)
            self._random_searcher._restore_from_state(state["random_searcher_state"])

    def set_profiler(self, profiler: Optional[SimpleProfiler]):
        self.profiler = profiler
        self.do_profile = profiler is not None

    @property
    def debug_log(self):
        return self._debug_log

    def _assign_random_searcher(self, random_seed=None):
        if self._random_searcher is None:
            # Used for initial random configs (if any)
            # We do not have to deal with ``points_to_evaluate``
            if random_seed is None:
                random_seed = self.random_state.randint(0, 2**32)
            self._random_searcher = RandomSearcher(
                self.hp_ranges.config_space_for_sampling,
                metric=self._metric,
                points_to_evaluate=[],
                random_seed=random_seed,
                debug_log=False,
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
      ``skip_local_optimization=True``).

    Note that the full logic of construction based on arguments is given in
    :mod:``syne_tune.optimizer.schedulers.searchers.gp_searcher_factory``. In
    particular, see
    :func:`~syne_tune.optimizer.schedulers.searchers.gp_searcher_factory.gp_fifo_searcher_defaults`
    for default values.

    Additional arguments on top of parent class
    :class:`~syne_tune.optimizer.schedulers.searchers.SearcherWithRandomSeed`:

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
        Defaults to ``syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults.DEFAULT_NUM_INITIAL_CANDIDATES``
    :type num_init_candidates: int, optional
    :param num_fantasy_samples: Number of samples drawn for fantasizing
        (latent target values for pending evaluations), defaults to 20
    :type num_fantasy_samples: int, optional
    :param no_fantasizing: If True, fantasizing is not done and pending
        evaluations are ignored. This may lead to loss of diversity in
        decisions
    :type no_fantasizing: Defaults to False
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
    :type transfer_learning_active_config_space: dict, optional
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
        config_space: dict,
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
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
            kwargs, *gp_fifo_searcher_defaults(), dict_name="search_options"
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
        from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

        assert isinstance(
            scheduler, FIFOScheduler
        ), "This searcher requires FIFOScheduler scheduler"
        super().configure_scheduler(scheduler)
        # Allow model factory to depend on ``scheduler`` as well
        model_factory = self.state_transformer.model_factory
        model_factory.configure_scheduler(scheduler)

    def register_pending(
        self, trial_id: str, config: Optional[dict] = None, milestone=None
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

    def _postprocess_config(self, config: dict) -> dict:
        return config

    def _get_config_modelbased(
        self, exclusion_candidates, **kwargs
    ) -> Optional[Configuration]:
        # Obtain current SurrogateModel from state transformer. Based on
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
            initial_candidates_generator=self.random_generator,
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
            # ``DebugLogWriter`` does not support batch selection right now,
            # must be switched off
            assert self.debug_log is None, (
                "``get_batch_configs`` does not support debug_log right now. "
                + "Please set ``debug_log=False`` in search_options argument "
                + "of scheduler, or create your searcher with ``debug_log=False``"
            )
            exclusion_candidates = self._get_exclusion_candidates(**kwargs)
            pick_random = True
            while pick_random and len(configs) < batch_size:
                config, pick_random = self._get_config_not_modelbased(
                    exclusion_candidates
                )
                if pick_random:
                    if config is not None:
                        configs.append(config)
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
                    model_factory = self.state_transformer._model_factory
                    if isinstance(model_factory, dict):
                        model_factory = model_factory[INTERNAL_METRIC_NAME]
                    # We need a copy of the state here, since
                    # ``pending_candidate_state_transformer`` modifies the state (it
                    # appends pending trials)
                    temporary_state = copy.deepcopy(self.state_transformer.state)
                    pending_candidate_state_transformer = ModelStateTransformer(
                        model_factory=model_factory,
                        init_state=temporary_state,
                        skip_optimization=AlwaysSkipPredicate(),
                    )
                bo_algorithm = BayesianOptimizationAlgorithm(
                    initial_candidates_generator=self.random_generator,
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
                _configs = bo_algorithm.next_candidates()
                configs.extend(self._postprocess_config(config) for config in _configs)
        return configs

    def evaluation_failed(self, trial_id: str):
        # Remove pending evaluation
        self.state_transformer.drop_pending_evaluation(trial_id)
        # Mark config as failed (which means it will be blacklisted in
        # future get_config calls)
        self.state_transformer.mark_trial_failed(trial_id)

    def _new_searcher_kwargs_for_clone(self) -> dict:
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
