from typing import Optional, List, Tuple, Dict, Any
import logging
import numpy as np
from dataclasses import dataclass

from syne_tune.optimizer.schedulers.random_seeds import RANDOM_SEED_UPPER_BOUND
from syne_tune.optimizer.schedulers.synchronous.dehb_bracket_manager import (
    DifferentialEvolutionHyperbandBracketManager,
)
from syne_tune.optimizer.schedulers.synchronous.hyperband_bracket import (
    SlotInRung,
)
from syne_tune.optimizer.schedulers.synchronous.hyperband import (
    SynchronousHyperbandCommon,
)
from syne_tune.optimizer.scheduler import TrialSuggestion, SchedulerDecision
from syne_tune.backend.trial_status import Trial
from syne_tune.config_space import cast_config_values
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    check_and_merge_defaults,
    Categorical,
    String,
    assert_no_invalid_options,
    Integer,
    Float,
    Boolean,
)
from syne_tune.optimizer.schedulers.searchers.legacy_searcher import (
    impute_points_to_evaluate,
)
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import ExclusionList
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)

logger = logging.getLogger(__name__)


_ARGUMENT_KEYS = {
    "searcher",
    "search_options",
    "metric",
    "mode",
    "points_to_evaluate",
    "random_seed",
    "max_resource_attr",
    "max_resource_level",
    "resource_attr",
    "mutation_factor",
    "crossover_probability",
    "support_pause_resume",
    "searcher_data",
}

_DEFAULT_OPTIONS = {
    "searcher": "random_encoded",
    "mode": "min",
    "resource_attr": "epoch",
    "mutation_factor": 0.5,
    "crossover_probability": 0.5,
    "support_pause_resume": True,
    "searcher_data": "rungs",
}

_CONSTRAINTS = {
    "metric": String(),
    "mode": Categorical(choices=("min", "max")),
    "random_seed": Integer(0, RANDOM_SEED_UPPER_BOUND),
    "max_resource_attr": String(),
    "max_resource_level": Integer(1, None),
    "resource_attr": String(),
    "mutation_factor": Float(lower=0, upper=1),
    "crossover_probability": Float(lower=0, upper=1),
    "support_pause_resume": Boolean(),
    "searcher_data": Categorical(("rungs", "all")),
}


@dataclass
class TrialInformation:
    """
    Information the scheduler maintains per trial.
    """

    encoded_config: np.ndarray
    level: int
    metric_val: Optional[float] = None


class ExtendedSlotInRung:
    """
    Extends :class:`SlotInRung` mostly for convenience
    """

    def __init__(self, bracket_id: int, slot_in_rung: SlotInRung):
        self.bracket_id = bracket_id
        self.rung_index = slot_in_rung.rung_index
        self.level = slot_in_rung.level
        self.slot_index = slot_in_rung.slot_index
        self.trial_id = slot_in_rung.trial_id
        self.metric_val = slot_in_rung.metric_val
        self.do_selection = False

    def slot_in_rung(self) -> SlotInRung:
        return SlotInRung(
            rung_index=self.rung_index,
            level=self.level,
            slot_index=self.slot_index,
            trial_id=self.trial_id,
            metric_val=self.metric_val,
        )


class DifferentialEvolutionHyperbandScheduler(SynchronousHyperbandCommon):
    """
    Differential Evolution Hyperband, as proposed in

        | DEHB: Evolutionary Hyperband for Scalable, Robust and Efficient Hyperparameter Optimization
        | Noor Awad, Neeratyoy Mallik, Frank Hutter
        | IJCAI 30 (2021), pages 2147-2153
        | https://arxiv.org/abs/2105.09821

    We implement DEHB as a variant of synchronous Hyperband, which may
    differ slightly from the implementation of the authors.
    Main differences to synchronous Hyperband:

    * In DEHB, trials are not paused and potentially promoted (except in the
      very first bracket). Therefore, checkpointing is not used (except in
      the very first bracket, if ``support_pause_resume`` is ``True``)
    * Only the initial configurations are drawn at random (or drawn from the
      searcher). Whenever possible, new configurations (in their internal
      encoding) are derived from earlier ones by way of differential evolution

    :param config_space: Configuration space for trial evaluation function
    :param rungs_first_bracket: Determines rung level systems for each
        bracket, see
        :class:`~syne_tune.optimizer.schedulers.synchronous.dehb_bracket_manager.DifferentialEvolutionHyperbandBracketManager`
    :param num_brackets_per_iteration: Number of brackets per iteration. The
        algorithm cycles through these brackets in one iteration. If not
        given, the maximum number is used (i.e., ``len(rungs_first_bracket)``)
    :param metric: Name of metric to optimize, key in result's obtained via
        :meth:`on_trial_result`
    :type metric: str
    :param searcher: Searcher for ``get_config`` decisions. Passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory` along
        with ``search_options`` and extra information. Supported values:
        :const:`~syne_tune.optimizer.schedulers.searchers.searcher_factory.SUPPORTED_SEARCHERS_HYPERBAND`.
        If ``searcher == "random_encoded"`` (default), the encoded configs are
        sampled directly, each entry independently from U([0, 1]).
        This distribution has higher entropy than for "random" if
        there are discrete hyperparameters in ``config_space``. Note that
        ``points_to_evaluate`` is still used in this case.
    :type searcher: str, optional
    :param search_options: Passed to
        :func:`~syne_tune.optimizer.schedulers.searchers.searcher_factory`.
        Note: If :code:`search_options["allow_duplicates"] == True`, then
        :meth:`suggest` may return a configuration more than once
    :type search_options: Dict[str, Any], optional
    :param mode: Mode to use for the metric given, can be "min" (default) or
        "max"
    :type mode: str, optional
    :param points_to_evaluate: List of configurations to be evaluated
        initially (in that order). Each config in the list can be partially
        specified, or even be an empty dict. For each hyperparameter not
        specified, the default value is determined using a midpoint heuristic.
        If None (default), this is mapped to ``[dict()]``, a single default config
        determined by the midpoint heuristic. If ``[]`` (empty list), no initial
        configurations are specified.
    :type points_to_evaluate: ``List[dict]``, optional
    :param random_seed: Master random seed. Generators used in the scheduler
        or searcher are seeded using
        :class:`~syne_tune.optimizer.schedulers.random_seeds.RandomSeedGenerator`.
        If not given, the master random seed is drawn at random here.
    :type random_seed: int, optional
    :param max_resource_attr: Key name in config for fixed attribute
        containing the maximum resource. If given, trials need not be
        stopped, which can run more efficiently.
    :type max_resource_attr: str, optional
    :param max_resource_level: Largest rung level, corresponds to ``max_t`` in
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. Must be positive
        int larger than ``grace_period``. If this is not given, it is inferred
        like in :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`. In
        particular, it is not needed if ``max_resource_attr`` is given.
    :type max_resource_level: int, optional
    :param resource_attr: Name of resource attribute in results obtained via
        :meth:`on_trial_result`. The type of resource must be int. Default to
        "epoch"
    :type resource_attr: str, optional
    :param mutation_factor: In :math:`(0, 1]`. Factor :math:`F` used in the rand/1
        mutation operation of DE. Default to 0.5
    :type mutation_factor: float, optional
    :param crossover_probability: In :math:`(0, 1)`. Probability :math:`p` used
        in crossover operation (child entries are chosen with probability
        :math:`p`). Defaults to 0.5
    :type crossover_probability: float, optional
    :param support_pause_resume: If ``True``, :meth:`_suggest` supports pause and
        resume in the first bracket (this is the default). If the objective
        supports checkpointing, this is made use of. Defaults to ``True``.
        Note: The resumed trial still gets assigned a new ``trial_id``, but it
        starts from the earlier checkpoint.
    :type support_pause_resume: bool, optional
    :param searcher_data: Relevant only if a model-based searcher is used.
        Example: For NN tuning and ``resource_attr == "epoch"``, we receive a
        result for each epoch, but not all epoch values are also rung levels.
        searcher_data determines which of these results are passed to the
        searcher. As a rule, the more data the searcher receives, the better
        its fit, but also the more expensive get_config may become. Choices:

        * "rungs" (default): Only results at rung levels. Cheapest
        * "all": All results. Most expensive

        Note: For a Gaussian additive learning curve surrogate model, this
        has to be set to "all".
    :type searcher_data: str, optional
    """

    MAX_RETRIES = 50

    def __init__(
        self,
        config_space: Dict[str, Any],
        rungs_first_bracket: List[Tuple[int, int]],
        num_brackets_per_iteration: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(config_space, **kwargs)
        self._create_internal(rungs_first_bracket, num_brackets_per_iteration, **kwargs)

    def _create_internal(
        self,
        rungs_first_bracket: List[Tuple[int, int]],
        num_brackets_per_iteration: Optional[int] = None,
        **kwargs,
    ):
        # Check values and impute default values
        assert_no_invalid_options(
            kwargs, _ARGUMENT_KEYS, name="DifferentialEvolutionHyperbandScheduler"
        )
        kwargs = check_and_merge_defaults(
            kwargs,
            set(),
            _DEFAULT_OPTIONS,
            _CONSTRAINTS,
            dict_name="scheduler_options",
        )
        self.mutation_factor = kwargs["mutation_factor"]
        self.crossover_probability = kwargs["crossover_probability"]
        self._support_pause_resume = kwargs["support_pause_resume"]
        search_options = self._create_internal_common(
            skip_searchers={"random_encoded"}, **kwargs
        )
        self._debug_log = None
        if self._searcher is None:
            if search_options.get("debug_log", True):
                self._debug_log = DebugLogPrinter()
            points_to_evaluate = kwargs.get("points_to_evaluate")
            self._points_to_evaluate = impute_points_to_evaluate(
                points_to_evaluate, self.config_space
            )
        else:
            self._debug_log = self._searcher.debug_log
            self._points_to_evaluate = None
        self._allow_duplicates = search_options.get("allow_duplicates", False)
        # Bracket manager
        self.bracket_manager = DifferentialEvolutionHyperbandBracketManager(
            rungs_first_bracket=rungs_first_bracket,
            mode=self.mode,
            num_brackets_per_iteration=num_brackets_per_iteration,
        )
        # Needed to convert encoded configs to configs
        self._hp_ranges = make_hyperparameter_ranges(self.config_space)
        self._excl_list = ExclusionList(self._hp_ranges)
        # PRNG for mutation and crossover random draws
        self.random_state = np.random.RandomState(self.random_seed_generator())
        # How often is selection skipped because target still pending?
        self.num_selection_skipped = 0
        # Maps ``trial_id`` to ``ext_slot``, as returned by ``bracket_manager.next_job``,
        # and required by ``bracket_manager.on_result``. Entries are removed once
        # passed to ``on_result``. Here, ``ext_slot`` is of type ``ExtendedSlotInRung``.
        self._trial_to_pending_slot = dict()
        # Maps trial_id to trial information (in particular, the encoded
        # config)
        self._trial_info = dict()
        # Maps level to list of trial_ids of all completed jobs (so that
        # metric values are available). This global "parent pool" is used
        # during mutations if the normal parent pool is too small
        self._global_parent_pool = {level: [] for _, level in rungs_first_bracket}
        self._rung_levels = [level for _, level in rungs_first_bracket]

    @property
    def rung_levels(self) -> List[int]:
        return self._rung_levels

    @property
    def num_brackets(self) -> int:
        return len(self.bracket_manager.bracket_rungs)

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        if self._excl_list.config_space_exhausted():
            logger.warning("All configurations in config space have been suggested")
            return None
        if self._debug_log is not None:
            if trial_id == 0:
                # This is printed at the start of the experiment. Cannot do this
                # at construction, because with ``RemoteLauncher`` this does not end
                # up in the right log
                parts = ["Rung systems for each bracket:"] + [
                    f"Bracket {bracket}: {rungs}"
                    for bracket, rungs in enumerate(self.bracket_manager.bracket_rungs)
                ]
                logger.info("\n".join(parts))
        # Ask bracket manager for job
        bracket_id, slot_in_rung = self.bracket_manager.next_job()
        ext_slot = ExtendedSlotInRung(bracket_id, slot_in_rung)
        if self._debug_log is not None:
            logger.info(
                f"trial_id {trial_id} for bracket {bracket_id}, level "
                f"{slot_in_rung.level}, rung index "
                + f"{slot_in_rung.rung_index}, slot {slot_in_rung.slot_index}"
            )
        is_base_rung = ext_slot.rung_index == 0  # Slot in base rung?
        encoded_config = None
        promoted_from_trial_id = None
        for next_config_iter in range(self.MAX_RETRIES):
            if next_config_iter < self.MAX_RETRIES / 2:
                draw_from_searcher = False
                if is_base_rung:
                    if bracket_id == 0:
                        draw_from_searcher = True
                    else:
                        parent_trial_id = (
                            self.bracket_manager.trial_id_from_parent_slot(
                                bracket_id=bracket_id,
                                level=ext_slot.level,
                                slot_index=ext_slot.slot_index,
                            )
                        )
                        draw_from_searcher = parent_trial_id is None
                if draw_from_searcher:
                    # At the very start, we draw configs from the searcher
                    encoded_config = self._encoded_config_from_searcher(trial_id)
                elif bracket_id == 0:
                    # First bracket, but not base rung. Promotion as in synchronous
                    # HB, but we assign new trial_id
                    (
                        encoded_config,
                        promoted_from_trial_id,
                    ) = self._encoded_config_by_promotion(ext_slot)
                else:
                    # Here, we can do DE (mutation, crossover)
                    encoded_config = self._extended_config_by_mutation_crossover(
                        ext_slot
                    )
            else:
                # Draw encoded config at random
                restore_searcher = self._searcher
                self._searcher = None
                encoded_config = self._encoded_config_from_searcher(trial_id)
                self._searcher = restore_searcher
            if encoded_config is None:
                break  # Searcher failed to return config
            if promoted_from_trial_id is not None:
                break  # Promotion is a config suggested before, that is OK
            _config = self._hp_ranges.from_ndarray(encoded_config)
            if not self._excl_list.contains(_config):
                break
            else:
                encoded_config = None
        if encoded_config is not None:
            if self._support_pause_resume and promoted_from_trial_id is not None:
                suggestion = self._promote_trial_and_make_suggestion(
                    trial_id=promoted_from_trial_id, ext_slot=ext_slot
                )
                if self._debug_log is not None:
                    logger.info(
                        f"trial_id {promoted_from_trial_id} resumes (milestone = "
                        f"{ext_slot.level})"
                    )
            else:
                suggestion = self._register_new_config_and_make_suggestion(
                    trial_id=trial_id, ext_slot=ext_slot, encoded_config=encoded_config
                )
                if self._debug_log is not None:
                    logger.info(
                        f"trial_id {trial_id} starts (milestone = {ext_slot.level})"
                    )
        else:
            # Searcher failed to return a config for a new trial_id. We report
            # the corresponding job as failed, so that in case the experiment
            # is continued, the bracket is not blocked with a slot which remains
            # pending forever
            logger.warning(
                "Searcher failed to suggest a configuration for new trial "
                f"{trial_id}. The corresponding slot is marked as failed."
            )
            self._report_as_failed(ext_slot)
            suggestion = None
        return suggestion

    def _encoded_config_from_searcher(self, trial_id: int) -> np.ndarray:
        config = None
        encoded_config = None
        if self.searcher is not None:
            if self._debug_log is not None:
                logger.info("Draw new config from searcher")
            config = self.searcher.get_config(trial_id=str(trial_id))
        else:
            # ``random_encoded`` internal searcher. Still, ``points_to_evaluate``
            # is used
            if self._points_to_evaluate:
                config = self._points_to_evaluate.pop(0)
            else:
                if self._debug_log is not None:
                    logger.info("Draw new encoded config uniformly at random")
                encoded_config = self.random_state.uniform(
                    low=0, high=1, size=self._hp_ranges.ndarray_size
                )
        if config is not None:
            encoded_config = self._hp_ranges.to_ndarray(config)
        return encoded_config

    def _encoded_config_by_promotion(
        self, ext_slot: ExtendedSlotInRung
    ) -> (Optional[np.ndarray], Optional[int]):
        parent_trial_id = self.bracket_manager.top_of_previous_rung(
            bracket_id=ext_slot.bracket_id, pos=ext_slot.slot_index
        )
        if parent_trial_id is not None:
            trial_info = self._trial_info[parent_trial_id]
            assert trial_info.metric_val is not None  # Sanity check
            if self._debug_log is not None:
                logger.info(
                    f"Promote config from trial_id {parent_trial_id}"
                    f", level {trial_info.level}"
                )
            encoded_config = trial_info.encoded_config
        else:
            # This can happen when all trials in the previous rung failed
            encoded_config = None
        return encoded_config, parent_trial_id

    def _extended_config_by_mutation_crossover(
        self, ext_slot: ExtendedSlotInRung
    ) -> np.ndarray:
        ext_slot.do_selection = True
        mutant = self._mutation(ext_slot)
        target_trial_id = self._get_target_trial_id(ext_slot)
        if self._debug_log is not None:
            logger.info(f"Target (cross-over): trial_id {target_trial_id}")
        return self._crossover(
            mutant=mutant,
            target=self._trial_info[target_trial_id].encoded_config,
        )

    def _draw_random_trial_id(self) -> int:
        return self.random_state.choice(list(self._trial_info.keys()))

    def _get_target_trial_id(self, ext_slot: ExtendedSlotInRung) -> Optional[int]:
        """
        The target trial_id is the trial_id in the parent slot. If this is None,
        a random existing trial_id is returned.
        """
        target_trial_id = self.bracket_manager.trial_id_from_parent_slot(
            bracket_id=ext_slot.bracket_id,
            level=ext_slot.level,
            slot_index=ext_slot.slot_index,
        )
        if target_trial_id is None:
            target_trial_id = self._draw_random_trial_id()
        return target_trial_id

    def _register_new_config_and_make_suggestion(
        self, trial_id: int, ext_slot: ExtendedSlotInRung, encoded_config: np.ndarray
    ) -> TrialSuggestion:
        # Register as pending
        self._trial_to_pending_slot[trial_id] = ext_slot
        # Register new trial_id
        self._trial_info[trial_id] = TrialInformation(
            encoded_config=encoded_config,
            level=ext_slot.level,
        )
        # Return new config
        config = self._hp_ranges.from_ndarray(encoded_config)
        if not self._allow_duplicates:
            self._excl_list.add(config)  # Should not be suggested again
        if self._debug_log is not None and self.searcher is None:
            self._debug_log.start_get_config("random", trial_id=trial_id)
            self._debug_log.set_final_config(config)
            self._debug_log.write_block()
        config = cast_config_values(config, self.config_space)
        if self.searcher is not None:
            self.searcher.register_pending(
                trial_id=str(trial_id), config=config, milestone=ext_slot.level
            )
        if self.max_resource_attr is not None:
            config = dict(config, **{self.max_resource_attr: ext_slot.level})
        return TrialSuggestion.start_suggestion(config=config)

    def _promote_trial_and_make_suggestion(
        self, trial_id: int, ext_slot: ExtendedSlotInRung
    ) -> TrialSuggestion:
        # Register as pending
        self._trial_to_pending_slot[trial_id] = ext_slot
        # Modify entry to new milestone level
        trial_info = self._trial_info.get(trial_id)
        assert (
            trial_info is not None
        ), f"Cannot promote trial_id {trial_id}, which is not registered"
        trial_info.level = ext_slot.level
        trial_info.metric_val = None
        if self.max_resource_attr is None:
            config = None
        else:
            config = cast_config_values(
                self._hp_ranges.from_ndarray(trial_info.encoded_config),
                self.config_space,
            )
            config = dict(config, **{self.max_resource_attr: ext_slot.level})
        return TrialSuggestion.resume_suggestion(trial_id=trial_id, config=config)

    def _report_as_failed(self, ext_slot: ExtendedSlotInRung):
        result_failed = ext_slot.slot_in_rung()
        result_failed.metric_val = np.NAN
        self.bracket_manager.on_result((ext_slot.bracket_id, result_failed))

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        trial_id = trial.trial_id
        if trial_id in self._trial_to_pending_slot:
            ext_slot = self._trial_to_pending_slot[trial_id]
            milestone = ext_slot.level
            metric_val, resource = self._extract_from_result(trial_id, result)
            trial_decision = SchedulerDecision.CONTINUE
            if resource >= milestone:
                assert resource == milestone, (
                    f"Trial trial_id {trial_id}: Obtained result for "
                    + f"resource = {resource}, but not for {milestone}. "
                    + "Training script must not skip rung levels!"
                )
                if self._debug_log is not None:
                    logger.info(
                        f"Trial trial_id {trial_id}: Reached milestone "
                        f"{milestone} with metric {metric_val:.3f}"
                    )
                self._record_new_metric_value(trial_id, milestone, metric_val)
                # Selection
                winner_trial_id = self._selection(trial_id, ext_slot, metric_val)
                # Return updated slot information to bracket
                self._return_slot_result_to_bracket(winner_trial_id, ext_slot)
                if self._support_pause_resume and ext_slot.bracket_id == 0:
                    trial_decision = SchedulerDecision.PAUSE
                else:
                    trial_decision = SchedulerDecision.STOP
                prev_level = self.bracket_manager.level_to_prev_level(
                    ext_slot.bracket_id, milestone
                )
                if resource > prev_level and self.searcher is not None:
                    config = self._hp_ranges.from_ndarray(
                        self._trial_info[trial_id].encoded_config
                    )
                    update = self.searcher_data == "all" or resource == milestone
                    self.searcher.on_trial_result(
                        trial_id=str(trial_id),
                        config=config,
                        result=result,
                        update=update,
                    )
        else:
            trial_decision = SchedulerDecision.STOP
            logger.warning(
                f"Received result for trial_id {trial_id}, which is not "
                f"pending. This result is not used:\n{result}"
            )

        return trial_decision

    def _extract_from_result(
        self, trial_id: int, result: Dict[str, Any]
    ) -> (float, int):
        metric_vals = []
        for name in (self.metric, self._resource_attr):
            assert name in result, (
                f"Result for trial_id {trial_id} does not contain " + f"'{name}' field"
            )
            metric_vals.append(result[name])
        return float(metric_vals[0]), int(metric_vals[1])

    def _record_new_metric_value(
        self, trial_id: int, milestone: int, metric_val: float
    ):
        # Record metric value
        trial_info = self._trial_info[trial_id]
        assert trial_info.level == milestone  # Sanity check
        trial_info.metric_val = metric_val
        # Update global parent pool
        self._global_parent_pool[milestone].append(trial_id)

    def _return_slot_result_to_bracket(
        self, winner_trial_id: int, ext_slot: ExtendedSlotInRung
    ):
        ext_slot.trial_id = winner_trial_id
        ext_slot.metric_val = self._trial_info[winner_trial_id].metric_val
        slot_in_rung = ext_slot.slot_in_rung()
        self.bracket_manager.on_result((ext_slot.bracket_id, slot_in_rung))

    def _mutation(self, ext_slot: ExtendedSlotInRung) -> np.ndarray:
        bracket_id = ext_slot.bracket_id
        level = ext_slot.level
        assert bracket_id > 0
        # Determine the parent pool, from which the 3 parents are sampled. If
        # this is too small (< 3), we also use the global parent pool for this
        # level (i.e., all completed trials), or even global parent pools at
        # all levels.
        orig_pool_size = self.bracket_manager.size_of_current_rung(bracket_id)
        pool_size = orig_pool_size
        global_pool = None
        if orig_pool_size < 3:
            global_pool = self._global_parent_pool[level]
            pool_size += len(global_pool)
            # If this is still too small, we add parent pools of other levels
            for _, other_level in reversed(self.bracket_manager.bracket_rungs[0]):
                if pool_size >= 3:
                    break
                if other_level != level:
                    extra_pool = self._global_parent_pool[other_level]
                    global_pool = global_pool + extra_pool
                    pool_size += len(extra_pool)
            # TODO: If this ever happens, have to do something else here. For
            # example, could pick trial_id's which are still pending
            assert pool_size >= 3, f"Cannot compose parent pool of size >= 3"
        # Sample 3 entries at random from parent pool
        positions = list(self.random_state.choice(pool_size, 3, replace=False))
        is_base_rung = ext_slot.rung_index == 0
        msg = None
        if self._debug_log is not None:
            from_str = "parent rung" if is_base_rung else "top of rung below"
            msg = f"Mutation: Sample parents from {from_str}: pool_size = {pool_size}"
            if orig_pool_size != pool_size:
                msg += f", orig_pool_size = {orig_pool_size}"
        parent_trial_ids = []
        for pos in positions:
            if pos >= orig_pool_size:
                trial_id = global_pool[pos - orig_pool_size]
            elif is_base_rung:
                trial_id = self.bracket_manager.trial_id_from_parent_slot(
                    bracket_id=bracket_id, level=level, slot_index=pos
                )
                if trial_id is None:
                    trial_id = self._draw_random_trial_id()
            else:
                trial_id = self.bracket_manager.top_of_previous_rung(
                    bracket_id=bracket_id, pos=pos
                )
            parent_trial_ids.append(trial_id)
        if self._debug_log is not None:
            msg += "\n" + str(parent_trial_ids)
            logger.info(msg)
        return self._de_mutation(parent_trial_ids)

    def _de_mutation(self, parent_trial_ids: List[int]) -> np.ndarray:
        """
        Corresponds to rand/1 mutation strategy of DE.
        """
        ec = [
            self._trial_info[trial_id].encoded_config for trial_id in parent_trial_ids
        ]
        mutant = (ec[1] - ec[2]) * self.mutation_factor + ec[0]
        # Entries which violate boundaries are resampled at random
        violations = np.where((mutant > 1) | (mutant < 0))[0]
        if len(violations) > 0:
            mutant[violations] = self.random_state.uniform(
                low=0.0, high=1.0, size=len(violations)
            )
        return mutant

    def _crossover(self, mutant: np.ndarray, target: np.ndarray) -> np.ndarray:
        # For any HP whose encoding has dimension > 1 (e.g., categorical), we
        # make sure not to cross-over inside the encoding
        num_hps = len(self._hp_ranges)
        hp_mask = self.random_state.rand(num_hps) < self.crossover_probability
        if not np.any(hp_mask):
            # Offspring must be different from target
            hp_mask[self.random_state.randint(0, num_hps)] = True
        cross_points = np.empty(self._hp_ranges.ndarray_size, dtype=bool)
        for (start, end), val in zip(self._hp_ranges.encoded_ranges.values(), hp_mask):
            cross_points[start:end] = val
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def _selection(
        self, trial_id: int, ext_slot: ExtendedSlotInRung, metric_val: float
    ) -> int:
        winner_trial_id = trial_id
        if ext_slot.do_selection:
            # Note: This can have changed since crossover
            target_trial_id = self._get_target_trial_id(ext_slot)
            if self._debug_log is not None:
                logger.info(f"Target (selection): trial_id {target_trial_id}")
            target_metric_val = self._trial_info[target_trial_id].metric_val
            if target_metric_val is not None:
                # Selection
                metric_sign = -1 if self.mode == "max" else 1
                if metric_sign * (metric_val - target_metric_val) >= 0:
                    winner_trial_id = target_trial_id
                if self._debug_log is not None:
                    logger.info(
                        f"Target metric = {target_metric_val:.3f}: "
                        f"winner_trial_id = {winner_trial_id}"
                    )
            else:
                # target has no metric value yet. This should not happen often
                logger.warning(
                    "Could not do selection because target metric "
                    f"value still pending (bracket_id = {ext_slot.bracket_id}"
                    f", trial_id = {trial_id}, "
                    f"target_trial_id = {target_trial_id})"
                )
                self.num_selection_skipped += 1
        return winner_trial_id

    def on_trial_error(self, trial: Trial):
        """
        Given the ``trial`` is currently pending, we send a result at its
        milestone for metric value NaN. Such trials are ranked after all others
        and will most likely not be promoted.

        """
        super().on_trial_error(trial)
        trial_id = trial.trial_id
        if trial_id in self._trial_to_pending_slot:
            ext_slot = self._trial_to_pending_slot[trial_id]
            self._report_as_failed(ext_slot)
            # A failed trial is not pending anymore
            del self._trial_to_pending_slot[trial_id]
        else:
            logger.warning(
                f"Trial trial_id {trial_id} not registered at pending: "
                "on_trial_error call is ignored"
            )

    def metric_names(self) -> List[str]:
        return [self.metric]

    def metric_mode(self) -> str:
        return self.mode
