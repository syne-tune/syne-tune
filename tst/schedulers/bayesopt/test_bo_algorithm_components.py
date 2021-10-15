import pytest

from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import \
    CandidateEvaluation, dictionarize_objective
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import \
    TuningJobState
from sagemaker_tune.search_space import uniform, choice, randint
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import tuples_to_configs


@pytest.fixture(scope='function')
def tuning_job_state():
    hp_ranges1 = make_hyperparameter_ranges({
        'a1_hp_1': uniform(-5.0, 5.0),
        'a1_hp_2': choice(['a', 'b', 'c'])})
    X1 = tuples_to_configs([(-3.0, 'a'), (-1.9, 'c'), (-3.5, 'a')], hp_ranges1)
    Y1 = [1.0, 2.0, 0.3]
    hp_ranges2 = make_hyperparameter_ranges({
        'a1_hp_1': uniform(-5.0, 5.0),
        'a1_hp_2': randint(-5, 5)})
    X2 = tuples_to_configs([(-1.9, -1), (-3.5, 3)], hp_ranges2)
    Y2 = [0.0, 2.0]
    return {
        'algo-1': TuningJobState(
            hp_ranges=hp_ranges1,
            candidate_evaluations=[
                CandidateEvaluation(x, dictionarize_objective(y))
                for x, y in zip(X1, Y1)],
            failed_candidates=[],
            pending_evaluations=[]),
        'algo-2': TuningJobState(
            hp_ranges=hp_ranges2,
            candidate_evaluations=[
                CandidateEvaluation(x, dictionarize_objective(y))
                for x, y in zip(X2, Y2)],
            failed_candidates=[],
            pending_evaluations=[]),
    }


@pytest.fixture(scope='function')
def tuning_job_sub_state():
    return TuningJobState(
        hp_ranges=make_hyperparameter_ranges(dict()),
        candidate_evaluations=[],
        failed_candidates=[],
        pending_evaluations=[])
