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
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    TransformerOutputModelFactory,
    ModelStateTransformer,
    StateForModelConverter,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    AcquisitionClassAndArgs,
    LocalOptimizer,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import (
    NoOptimization,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common import (
    ExclusionList,
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
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler

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
        self.set_profiler(model_factory_main.profiler)
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

        return ExclusionList(
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

    def set_profiler(self, profiler: Optional[SimpleProfiler]):
        self.profiler = profiler
        self.do_profile = profiler is not None

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
