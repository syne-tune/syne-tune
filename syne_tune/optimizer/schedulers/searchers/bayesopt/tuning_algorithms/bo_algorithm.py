from typing import List, Tuple, Iterator, Optional
import logging
from dataclasses import dataclass
import numpy as np
import itertools

from syne_tune.optimizer.schedulers.searchers.utils.common import Configuration
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer import (
    ModelStateTransformer,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes import (
    NextCandidatesAlgorithm,
    ScoringFunction,
    LocalOptimizer,
    Predictor,
    CandidateGenerator,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components import (
    LBFGSOptimizeAcquisition,
    generate_unique_candidates,
    DuplicateDetector,
)
from syne_tune.optimizer.schedulers.searchers.utils.exclusion_list import ExclusionList
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)

logger = logging.getLogger(__name__)


@dataclass
class BayesianOptimizationAlgorithm(NextCandidatesAlgorithm):
    """
    Core logic of the Bayesian optimization algorithm

    :param initial_candidates_generator: generator of candidates
    :param initial_scoring_function: scoring function used to rank the initial
        candidates.
        Note: If a batch is selected in one go (``num_requested_candidates > 1``,
        ``greedy_batch_selection == False``), this function should encourage
        diversity among its top scorers. In general, greedy batch selection
        is recommended.
    :param num_initial_candidates: how many initial candidates to generate, if
        possible
    :param local_optimizer: local optimizer which starts from score minimizer.
        If a batch is selected in one go (not greedily), then local
        optimizations are started from the top ``num_requested_candidates`` ranked
        candidates (after scoring)
    :param pending_candidate_state_transformer: Once a candidate is selected, it
        becomes pending, and the state is transformed by appending information.
        This is done by the transformer.
        This is object is needed only if :meth:`next_candidates` goes through
        more than one outer iterations (i.e., if ``greedy_batch_selection == True``
        and ``num_requested_candidates > 1``. Otherwise, None can be passed here.
        Note: Model updates (by the state transformer) for batch candidates beyond
        the first do not involve fitting hyperparameters, so they are usually
        cheap.
    :param exclusion_candidates: Set of candidates that should not be returned,
        because they are already labeled, currently pending, or have failed
    :param num_requested_candidates: number of candidates to return
    :param greedy_batch_selection: If True and ``num_requested_candidates > 1``, we
        generate, order, and locally optimize for each single candidate to be
        selected. Otherwise, this is done just once, and
        ``num_requested_candidates`` are extracted in one go.
        Note: If this is True, ``pending_candidate_state_transformer`` is needed.
    :param duplicate_detector: used to make sure no candidates equal to already
        evaluated ones is returned
    :param num_initial_candidates_for_batch: This is used only if
        ``num_requested_candidates > 1`` and ``greedy_batch_selection == True``. In
        this case, ``num_initial_candidates_for_batch`` overrides
        ``num_initial_candidates`` when selecting all but the first candidate for
        the batch. Typically, ``num_initial_candidates`` is larger than
        ``num_initial_candidates_for_batch`` in this case, which speeds up
        selecting large batches, but still select the first candidate
        thoroughly
    :param sample_unique_candidates: If ``True``, we check that initial candidates
        sampled at random are unique and disjoint from the exclusion list.
        This can be expensive. Defaults to ``False``
    :param debug_log: If a
        :class:`~syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log.DebugLogPrinter`
        object is passed here, it is used to write log messages
    """

    initial_candidates_generator: CandidateGenerator
    initial_candidates_scorer: ScoringFunction
    num_initial_candidates: int
    local_optimizer: LocalOptimizer
    pending_candidate_state_transformer: Optional[ModelStateTransformer]
    exclusion_candidates: ExclusionList
    num_requested_candidates: int
    greedy_batch_selection: bool
    duplicate_detector: DuplicateDetector
    num_initial_candidates_for_batch: Optional[int] = None
    sample_unique_candidates: bool = False
    debug_log: Optional[DebugLogPrinter] = None

    # Note: For greedy batch selection (num_outer_iterations > 1), the
    # underlying ``Predictor`` changes with each new pending candidate. The
    # model changes are managed by ``pending_candidate_state_transformer``. The
    # model has to be passed to both ``initial_candidates_scorer`` and
    # ``local_optimizer``.
    def next_candidates(self) -> List[Configuration]:
        if self.greedy_batch_selection:
            # Select batch greedily, one candidate at a time, updating the
            # model in between
            num_outer_iterations = self.num_requested_candidates
            num_inner_candidates = 1
        else:
            # Select batch in one go
            num_outer_iterations = 1
            num_inner_candidates = self.num_requested_candidates
        next_trial_id = None
        if num_outer_iterations > 1:
            assert (
                self.pending_candidate_state_transformer
            ), "Need pending_candidate_state_transformer for greedy batch selection"
            # For greedy batch selection, we need to assign new trial_id's to
            # configs included into the batch, in order to update the state
            # maintained in ``pending_candidate_state_transformer``.
            # This is just to make batch suggestion work: neither the state
            # nor these trial_id's are used in the future.
            # Note: This code also works if trial_id's are arbitrary strings.
            # It guarantees that ``str(next_trial_id + i)`` is not equal to an
            # existing trial_id for all i >= 0.
            next_trial_id = 0
            for (
                trial_id
            ) in self.pending_candidate_state_transformer.state.config_for_trial.keys():
                try:
                    next_trial_id = max(next_trial_id, int(trial_id))
                except ValueError:
                    pass
            next_trial_id += 1
        candidates = []
        just_added = True
        predictor = None  # Predictor, if num_outer_iterations > 1
        for outer_iter in range(num_outer_iterations):
            if just_added:
                if self.exclusion_candidates.config_space_exhausted():
                    logger.warning(
                        "All entries of finite config space (size "
                        + f"{self.exclusion_candidates.configspace_size}) have been selected. Returning "
                        + f"{len(candidates)} configs instead of {self.num_requested_candidates}"
                    )
                    break
                just_added = False
            if (
                self.num_initial_candidates_for_batch is not None
                and self.greedy_batch_selection
                and outer_iter > 0
            ):
                num_initial_candidates = self.num_initial_candidates_for_batch
            else:
                num_initial_candidates = self.num_initial_candidates
            inner_candidates = self._get_next_candidates(
                num_inner_candidates,
                predictor=predictor,
                num_initial_candidates=num_initial_candidates,
            )
            candidates.extend(inner_candidates)
            if outer_iter < num_outer_iterations - 1 and len(inner_candidates) > 0:
                just_added = True
                # This is not the last outer iteration
                for cand in inner_candidates:
                    self.exclusion_candidates.add(cand)
                # State transformer is used to produce new model
                # Note: We suppress fit_hyperpars for models obtained during
                # batch selection
                for candidate in inner_candidates:
                    self.pending_candidate_state_transformer.append_trial(
                        trial_id=str(next_trial_id), config=candidate
                    )
                    next_trial_id += 1
                predictor = self.pending_candidate_state_transformer.fit(
                    skip_optimization=True
                )
            if (
                len(inner_candidates) < num_inner_candidates
                and len(candidates) < self.num_requested_candidates
            ):
                logger.warning(
                    "All entries of finite config space (size "
                    + f"{self.exclusion_candidates.configspace_size}) have been selected. Returning "
                    + f"{len(candidates)} configs instead of {self.num_requested_candidates}"
                )
                break

        return candidates

    def _get_next_candidates(
        self,
        num_candidates: int,
        predictor: Optional[Predictor],
        num_initial_candidates: Optional[int] = None,
    ):
        if num_initial_candidates is None:
            num_initial_candidates = self.num_initial_candidates
        # generate a random candidates among which to pick the ones to be
        # locally optimized
        logger.info(
            f"BayesOpt Algorithm: Generating {num_initial_candidates} "
            "initial candidates."
        )
        if self.sample_unique_candidates:
            # This can be expensive, depending on what type Candidate is
            initial_candidates = generate_unique_candidates(
                self.initial_candidates_generator,
                num_initial_candidates,
                self.exclusion_candidates,
            )
        else:
            # Will not return candidates in ``exclusion_candidates``, but there
            # can be duplicates
            initial_candidates = (
                self.initial_candidates_generator.generate_candidates_en_bulk(
                    num_initial_candidates, exclusion_list=self.exclusion_candidates
                )
            )
        logger.info("BayesOpt Algorithm: Scoring (and reordering) candidates.")
        if self.debug_log is not None:
            candidates_and_scores = _order_candidates(
                candidates=initial_candidates,
                scoring_function=self.initial_candidates_scorer,
                predictor=predictor,
                with_scores=True,
            )
            initial_candidates = [cand for score, cand in candidates_and_scores]
            config = initial_candidates[0]
            top_scores = np.array([x for x, _ in candidates_and_scores[:5]])
            self.debug_log.set_init_config(config, top_scores)
        else:
            initial_candidates = _order_candidates(
                candidates=initial_candidates,
                scoring_function=self.initial_candidates_scorer,
                predictor=predictor,
            )
        candidates_with_optimization = _lazily_locally_optimize(
            candidates=initial_candidates,
            local_optimizer=self.local_optimizer,
            hp_ranges=self.exclusion_candidates.hp_ranges,
            predictor=predictor,
        )
        logger.info("BayesOpt Algorithm: Selecting final set of candidates.")
        if self.debug_log is not None and isinstance(
            self.local_optimizer, LBFGSOptimizeAcquisition
        ):
            # We would like to get num_evaluations from the first run (usually
            # the only one). This requires peeking at the first entry of the
            # iterator
            peek = candidates_with_optimization.__next__()
            self.debug_log.set_num_evaluations(self.local_optimizer.num_evaluations)
            candidates_with_optimization = itertools.chain(
                [peek], candidates_with_optimization
            )
        candidates = _pick_from_locally_optimized(
            candidates_with_optimization=candidates_with_optimization,
            exclusion_candidates=self.exclusion_candidates,
            num_candidates=num_candidates,
            duplicate_detector=self.duplicate_detector,
        )
        return candidates


def _order_candidates(
    candidates: List[Configuration],
    scoring_function: ScoringFunction,
    predictor: Optional[Predictor],
    with_scores: bool = False,
):
    if len(candidates) == 0:
        return []
    # scored in batch as this can be more efficient
    scores = scoring_function.score(candidates, predictor=predictor)
    sorted_list = sorted(zip(scores, candidates), key=lambda x: x[0])
    if with_scores:
        return sorted_list
    else:
        return [cand for score, cand in sorted_list]


def _lazily_locally_optimize(
    candidates: List[Configuration],
    local_optimizer: LocalOptimizer,
    hp_ranges: HyperparameterRanges,
    predictor: Optional[Predictor],
) -> Iterator[Tuple[Configuration, Configuration]]:
    """
    Due to local deduplication we do not know in advance how many candidates
    we have to locally optimize, hence this helper to create a lazy generator
    of locally optimized candidates.
    Note that ``candidates`` may contain duplicates, but such are skipped here.
    """
    considered_already = ExclusionList(hp_ranges)
    for cand in candidates:
        if not considered_already.contains(cand):
            considered_already.add(cand)
            yield cand, local_optimizer.optimize(cand, predictor=predictor)


# Note: If ``duplicate_detector`` is at least :class:`DuplicateDetectorIdentical`,
# it will filter out candidates in exclusion_candidates here. Such can in
# principle arise if ``sample_unique_candidates == False``. This does not work
# if ``duplicate_detector`` is of type :class:`DuplicateDetectorNoDetection`.
def _pick_from_locally_optimized(
    candidates_with_optimization: Iterator[Tuple[Configuration, Configuration]],
    exclusion_candidates: ExclusionList,
    num_candidates: int,
    duplicate_detector: DuplicateDetector,
) -> List[Configuration]:
    updated_excludelist = exclusion_candidates.copy()
    result = []
    for original_candidate, optimized_candidate in candidates_with_optimization:
        insert_candidate = None
        optimized_is_duplicate = duplicate_detector.contains(
            updated_excludelist, optimized_candidate
        )
        if optimized_is_duplicate:
            # in the unlikely case that the optimized candidate ended at a
            # place that caused a duplicate we try to return the original instead
            original_also_duplicate = duplicate_detector.contains(
                updated_excludelist, original_candidate
            )
            if not original_also_duplicate:
                insert_candidate = original_candidate
        else:
            insert_candidate = optimized_candidate
        if insert_candidate is not None:
            result.append(insert_candidate)
            updated_excludelist.add(insert_candidate)
        if len(result) == num_candidates:
            break

    return result
