import numpy as np

from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state \
    import TuningJobState, CandidateEvaluation
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model \
    import get_internal_candidate_evaluations
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import dictionarize_objective, \
    INTERNAL_METRIC_NAME
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import dimensionality_and_warping_ranges
from sagemaker_tune.search_space import uniform, randint, choice, loguniform
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges


def test_get_internal_candidate_evaluations():
    """we do not test the case with no evaluations, as it is assumed
    that there will be always some evaluations generated in the beginning
    of the BO loop."""

    hp_ranges = make_hyperparameter_ranges({
        'a': randint(0, 10),
        'b': uniform(0.0, 10.0),
        'c': choice(['X', 'Y'])})
    candidate_evaluations = [
        CandidateEvaluation(hp_ranges.tuple_to_config((2, 3.3, 'X')),
                            dictionarize_objective(5.3)),
        CandidateEvaluation(hp_ranges.tuple_to_config((1, 9.9, 'Y')),
                            dictionarize_objective(10.9)),
        CandidateEvaluation(hp_ranges.tuple_to_config((7, 6.1, 'X')),
                            dictionarize_objective(13.1))]

    state = TuningJobState(
        hp_ranges=hp_ranges,
        candidate_evaluations=candidate_evaluations,
        failed_candidates=[candidate_evaluations[0].candidate],  # these should be ignored by the model
        pending_evaluations=[]
    )

    result = get_internal_candidate_evaluations(
        state, INTERNAL_METRIC_NAME, normalize_targets=True,
        num_fantasy_samples=20)

    assert len(result.features.shape) == 2, "Input should be a matrix"
    assert len(result.targets.shape) == 2, "Output should be a matrix"

    assert result.features.shape[0] == len(candidate_evaluations)
    assert result.targets.shape[-1] == 1, \
        "Only single output value per row is suppored"

    assert np.abs(np.mean(result.targets)) < 1e-8, \
        "Mean of the normalized outputs is not 0.0"
    assert np.abs(np.std(result.targets) - 1.0) < 1e-8, \
        "Std. of the normalized outputs is not 1.0"

    np.testing.assert_almost_equal(result.mean, 9.766666666666666)
    np.testing.assert_almost_equal(result.std, 3.283629428273267)


def test_dimensionality_and_warping_ranges():
    hp_ranges = make_hyperparameter_ranges({
        'a': choice(['X', 'Y']),
        'b': loguniform(0.1, 10.0),
        'c': choice(['a', 'b', 'c']),
        'd': uniform(0.0, 10.0),
        'e': choice(['X', 'Y'])})

    dim, warping_ranges = dimensionality_and_warping_ranges(hp_ranges)
    assert dim == 9
    assert warping_ranges == {2: (0.0, 1.0), 6: (0.0, 1.0)}
