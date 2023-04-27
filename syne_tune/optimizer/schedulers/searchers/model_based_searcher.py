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
import time
from typing import Optional, Type, Dict, Any, List
import logging
import numpy as np
import copy

from syne_tune.optimizer.schedulers.searchers import (
    StochasticSearcher,
    RandomSearcher,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    INTERNAL_METRIC_NAME,
    MetricValues,
    INTERNAL_COST_NAME,
    TrialEvaluations,
    dictionarize_objective,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_skipopt import (
    SkipOptimizationPredicate,
    AlwaysSkipPredicate,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    TransformerOutputModelFactory,
    ModelStateTransformer,
    StateForModelConverter,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    AcquisitionClassAndArgs,
    LocalOptimizer,
    ScoringFunction,
    SurrogateOutputModel,
    unwrap_acquisition_class_and_kwargs,
    CandidateGenerator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm import (
    BayesianOptimizationAlgorithm,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import (
    NoOptimization,
    RandomStatefulCandidateGenerator,
    RandomFromSetCandidateGenerator,
    DuplicateDetectorIdentical,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import (
    IndependentThompsonSampling,
)
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import (
    ExclusionList,
    ExclusionListFromState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults import (
    DEFAULT_NUM_INITIAL_CANDIDATES,
    DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
    DEFAULT_LOCAL_OPTIMIZER_CLASS,
)
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import (
    DEFAULT_INITIAL_SCORING,
    SUPPORTED_INITIAL_SCORING,
    MapReward,
    encode_state,
)
from syne_tune.optimizer.schedulers.searchers.utils import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    ConfigurationFilter,
    Configuration,
)

logger = logging.getLogger(__name__)


GET_CONFIG_RANDOM_RETRIES = 50


def check_initial_candidates_scorer(initial_scoring: Optional[str]) -> str:
    if initial_scoring is None:
        return DEFAULT_INITIAL_SCORING
    else:
        assert (
            initial_scoring in SUPPORTED_INITIAL_SCORING
        ), "initial_scoring = '{}' is not supported".format(initial_scoring)
        return initial_scoring


class ModelBasedSearcher(StochasticSearcher):
    """Common code for surrogate model based searchers

    If ``num_initial_random_choices > 0``, initial configurations are drawn using
    an internal :class:`~syne_tune.optimizer.schedulers.searchers.RandomSearcher`
    object, which is created in :meth:`_assign_random_searcher`. This internal
    random searcher shares :attr:`random_state` with the searcher here. This ensures
    that if ``ModelBasedSearcher`` and ``RandomSearcher`` objects are created with
    the same ``random_seed`` and ``points_to_evaluate`` argument, initial
    configurations are identical until :meth:`_get_config_modelbased` kicks in.

    Note that this works because :attr:`random_state` is only used in the internal
    random searcher until meth:`_get_config_modelbased` is first called.
    """

    def _create_internal(
        self,
        hp_ranges: HyperparameterRanges,
        model_factory: TransformerOutputModelFactory,
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
        state_converter: Optional[StateForModelConverter] = None,
        allow_duplicates: bool = False,
        restrict_configurations: List[Dict[str, Any]] = None,
    ):
        self.hp_ranges = hp_ranges
        self.num_initial_candidates = num_initial_candidates
        self.num_initial_random_choices = num_initial_random_choices
        self.map_reward = map_reward
        if restrict_configurations is not None:
            restrict_configurations = self._filter_points_to_evaluate(
                restrict_configurations, hp_ranges, allow_duplicates
            )
            if not skip_local_optimization:
                logger.warning(
                    "If restrict_configurations is given, need to have skip_local_optimization == True"
                )
                skip_local_optimization = True
        self._restrict_configurations = restrict_configurations
        if skip_local_optimization:
            self.local_minimizer_class = NoOptimization
        else:
            self.local_minimizer_class = (
                DEFAULT_LOCAL_OPTIMIZER_CLASS
                if local_minimizer_class is None
                else local_minimizer_class
            )
        self.acquisition_class = acquisition_class
        if isinstance(model_factory, dict):
            model_factory_main = model_factory[INTERNAL_METRIC_NAME]
        else:
            model_factory_main = model_factory
        self._debug_log = model_factory_main.debug_log
        self.initial_scoring = check_initial_candidates_scorer(initial_scoring)
        self.skip_local_optimization = skip_local_optimization
        # Create state transformer
        # Initial state is empty (note that the state is mutable).
        # If there is a state converter, it uses the same random state as the searcher
        # here
        if state_converter is not None:
            state_converter.set_random_state(self.random_state)
        if init_state is None:
            init_state = TuningJobState.empty_state(self._hp_ranges_in_state())
        self.state_transformer = ModelStateTransformer(
            model_factory=model_factory,
            init_state=init_state,
            skip_optimization=skip_optimization,
            state_converter=state_converter,
        )
        self._cost_attr = cost_attr
        self._resource_attr = resource_attr
        self._filter_observed_data = filter_observed_data
        self._allow_duplicates = allow_duplicates
        self._random_searcher = None
        # Tracks the cumulative time spent in ``get_config`` calls
        self.cumulative_get_config_time = 0
        if self._debug_log is not None:
            msg_parts = [
                "[ModelBasedSearcher._create_internal]",
                f"- acquisition_class = {acquisition_class}",
                f"- local_minimizer_class = {self.local_minimizer_class}",
                f"- num_initial_candidates = {num_initial_candidates}",
                f"- num_initial_random_choices = {num_initial_random_choices}",
                f"- initial_scoring = {initial_scoring}",
                f"- allow_duplicates = {self._allow_duplicates}",
            ]
            logger.info("\n".join(msg_parts))

    def _copy_kwargs_to_kwargs_int(
        self, kwargs_int: Dict[str, Any], kwargs: Dict[str, Any]
    ):
        """Copies extra arguments not dealt with by ``gp_fifo_searcher_factory``

        :param kwargs_int: Output of factory, to be passed to ``searcher_factory``
        :param kwargs: Input arguments
        """
        # Extra arguments not parsed in factory
        for k in (
            "init_state",
            "local_minimizer_class",
            "cost_attr",
            "resource_attr",
            "restrict_configurations",
        ):
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

    def _metric_val_update(
        self, crit_val: float, result: Dict[str, Any]
    ) -> MetricValues:
        return crit_val

    def on_trial_result(
        self,
        trial_id: str,
        config: Dict[str, Any],
        result: Dict[str, Any],
        update: bool,
    ):
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

    def _trial_id_string(self, trial_id: str, result: Dict[str, Any]):
        """
        For multi-fidelity, we also want to output the resource level
        """
        return trial_id

    def _update(self, trial_id: str, config: Dict[str, Any], result: Dict[str, Any]):
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

    def _get_exclusion_candidates(self, skip_observed: bool = False) -> ExclusionList:
        def skip_all(config: Configuration) -> bool:
            return False

        return ExclusionListFromState(
            self.state_transformer.state,
            filter_observed_data=skip_all
            if skip_observed
            else self._filter_observed_data,
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
            for _ in range(GET_CONFIG_RANDOM_RETRIES):
                _config = self._random_searcher.get_config()
                if _config is None:
                    # If ``RandomSearcher`` returns no config at all, the
                    # search space is exhausted
                    break
                if not exclusion_candidates.contains(_config):
                    config = _config
                    break
            # ``_random_searcher`` modified ``restrict_configurations``
            if self._restrict_configurations is not None:
                self._restrict_configurations = (
                    self._random_searcher._restrict_configurations.copy()
                )
        return config, pick_random

    def get_config(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Runs Bayesian optimization in order to suggest the next config to evaluate.

        :return: Next config to evaluate at
        """
        start_time = time.time()
        state = self.state_transformer.state
        # Initial configs come from ``points_to_evaluate`` or are drawn at random
        # We use ``exclusion_candidates`` even if ``allow_duplicates == True``, in order
        # to count how many unique configs have been suggested
        exclusion_candidates = self._get_exclusion_candidates()
        config, pick_random = self._get_config_not_modelbased(exclusion_candidates)
        if self.debug_log is not None:
            trial_id = kwargs.get("trial_id")
            self.debug_log.start_get_config(
                "random" if pick_random else "BO", trial_id=trial_id
            )
        if not pick_random:
            # Model-based decision
            if self._allow_duplicates or (
                not exclusion_candidates.config_space_exhausted()
            ):
                # Even if ``allow_duplicates == True``, we exclude configs which are
                # pending or failed
                if self._allow_duplicates:
                    excl_cands = self._get_exclusion_candidates(skip_observed=True)
                else:
                    excl_cands = exclusion_candidates
                config = self._get_config_modelbased(
                    exclusion_candidates=excl_cands, **kwargs
                )

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
        self.cumulative_get_config_time += time.time() - start_time

        return config

    def dataset_size(self):
        return self.state_transformer.state.num_observed_cases()

    def model_parameters(self):
        return self.state_transformer.get_params()

    def set_params(self, param_dict):
        self.state_transformer.set_params(param_dict)

    def get_state(self) -> Dict[str, Any]:
        """
        The mutable state consists of the GP model parameters, the
        ``TuningJobState``, and the ``skip_optimization`` predicate (which can have a
        mutable state).
        We assume that ``skip_optimization`` can be pickled.

        Note that we do not have to store the state of :attr:`_random_searcher`,
        since this internal searcher shares its ``random_state`` with the searcher
        here.
        """
        state = dict(
            super().get_state(),
            model_params=self.model_parameters(),
            state=encode_state(self.state_transformer.state),
            skip_optimization=self.state_transformer.skip_optimization,
        )
        if self._restrict_configurations is not None:
            state["restrict_configurations"] = self._restrict_configurations
        return state

    def _restore_from_state(self, state: Dict[str, Any]):
        super()._restore_from_state(state)
        self.state_transformer.set_params(state["model_params"])
        self._restrict_configurations = state.get("restrict_configurations")
        # The internal random searcher is generated once needed, and it shares its
        # ``random_state`` with this searcher here
        self._random_searcher = None

    @property
    def debug_log(self):
        return self._debug_log

    def _assign_random_searcher(self):
        """
        Assigns :attr:`_random_searcher` if not already done. This internal searcher
        is sharing :attr:`random_state` with the searcher here, see header comments.
        """
        if self._random_searcher is None:
            # Used for initial random configs (if any)
            # We do not have to deal with ``points_to_evaluate``
            self._random_searcher = RandomSearcher(
                self.hp_ranges.config_space_for_sampling,
                metric=self._metric,
                points_to_evaluate=[],
                random_seed=0,
                debug_log=False,
                allow_duplicates=self._allow_duplicates,
                restrict_configurations=self._restrict_configurations,
            )
            self._random_searcher.set_random_state(self.random_state)


def create_initial_candidates_scorer(
    initial_scoring: str,
    model: SurrogateOutputModel,
    acquisition_class: AcquisitionClassAndArgs,
    random_state: np.random.RandomState,
    active_metric: str = INTERNAL_METRIC_NAME,
) -> ScoringFunction:
    if initial_scoring == "thompson_indep":
        if isinstance(model, dict):
            assert active_metric in model
            model = model[active_metric]
        return IndependentThompsonSampling(model, random_state=random_state)
    else:
        acquisition_class, acquisition_kwargs = unwrap_acquisition_class_and_kwargs(
            acquisition_class
        )
        return acquisition_class(
            model, active_metric=active_metric, **acquisition_kwargs
        )


class BayesianOptimizationSearcher(ModelBasedSearcher):
    """Common Code for searchers using Bayesian optimization

    We implement Bayesian optimization, based on a model factory which
    parameterizes the state transformer. This implementation works with
    any type of surrogate model and acquisition function, which are
    compatible with each other.

    The following happens in :meth:`get_config`:

    * For the first ``num_init_random`` calls, a config is drawn at random
      (after ``points_to_evaluate``, which are included in the ``num_init_random``
      initial ones). Afterwards, Bayesian optimization is used, unless there
      are no finished evaluations yet (a surrogate model cannot be used with no
      data at all)
    * For BO, model hyperparameter are refit first. This step can be skipped
      (see ``opt_skip_*`` parameters).
    * Next, the BO decision is made based on
      :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm.BayesianOptimizationAlgorithm`.
      This involves sampling `num_init_candidates`` configs are sampled at
      random, ranking them with a scoring function (``initial_scoring``), and
      finally runing local optimization starting from the top scoring config.
    """

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
        state = self.state_transformer.state
        if not state.is_pending(trial_id):
            assert not state.is_labeled(trial_id), (
                f"Trial trial_id = {trial_id} is already labeled, so cannot "
                "be pending"
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
        # Note: Asking for the model triggers the posterior computation
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
