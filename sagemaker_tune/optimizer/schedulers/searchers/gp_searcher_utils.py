from collections import Counter
from typing import NamedTuple, Callable

from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state \
    import TuningJobState
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import CandidateEvaluation, PendingEvaluation, INTERNAL_METRIC_NAME
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges


class MapReward(NamedTuple):
    forward: Callable[[float], float]
    reverse: Callable[[float], float]

    def __call__(self, x: float) -> float:
        return self.forward(x)


def map_reward_const_minus_x(const=1.0) -> MapReward:
    """
    Factory for map_reward argument in GPMultiFidelitySearcher.
    """
    def const_minus_x(x):
        return const - x

    return MapReward(forward=const_minus_x, reverse=const_minus_x)


SUPPORTED_INITIAL_SCORING = {
    'thompson_indep',
    'acq_func'}


DEFAULT_INITIAL_SCORING = 'thompson_indep'


def encode_state(state: TuningJobState) -> dict:
    candidate_evaluations = [
        {'candidate': x.candidate,
         'metrics': x.metrics}
        for x in state.candidate_evaluations]
    enc_state = {
        'candidate_evaluations': candidate_evaluations,
        'failed_candidates': state.failed_candidates,
        'pending_candidates': state.pending_candidates}
    return enc_state


def decode_state(enc_state: dict, hp_ranges: HyperparameterRanges) \
        -> TuningJobState:
    candidate_evaluations = [
        CandidateEvaluation(x['candidate'], x['metrics'])
        for x in enc_state['candidate_evaluations']]
    pending_evaluations = [
        PendingEvaluation(x) for x in enc_state['pending_candidates']]
    return TuningJobState(
        hp_ranges=hp_ranges,
        candidate_evaluations=candidate_evaluations,
        failed_candidates=enc_state['failed_candidates'],
        pending_evaluations=pending_evaluations)


class ResourceForAcquisitionMap(object):
    def __call__(self, state: TuningJobState, **kwargs) -> int:
        raise NotImplementedError()


class ResourceForAcquisitionBOHB(ResourceForAcquisitionMap):
    def __init__(
            self, threshold: int, active_metric: str = INTERNAL_METRIC_NAME):
        self.threshold = threshold
        self.active_metric = active_metric

    def __call__(self, state: TuningJobState, **kwargs) -> int:
        assert state.num_observed_cases(self.active_metric) > 0, \
            f"state must have data for metric {self.active_metric}"
        all_resources = []
        for cand_eval in state.candidate_evaluations:
            all_resources += [
                int(r) for r in cand_eval.metrics[self.active_metric].keys()]
        histogram = Counter(all_resources)
        return self._max_at_least_threshold(histogram)

    def _max_at_least_threshold(self, counter: Counter) -> int:
        """
        Get largest key of `counter` whose value is at least `threshold`.

        :param counter: dict with keys that support comparison operators
        :return: largest key of `counter`
        """
        return max(filter(
            lambda r: counter[r] >= self.threshold, counter.keys()),
            default=min(counter.keys()))


class ResourceForAcquisitionFirstMilestone(ResourceForAcquisitionMap):
    def __call__(self, state: TuningJobState, **kwargs) -> int:
        assert 'milestone' in kwargs, \
            "Need the first milestone to be attained by the new config " +\
            "passed as kwargs['milestone']. Use a scheduler which does " +\
            "that (e.g., HyperbandScheduler)"
        return kwargs['milestone']
