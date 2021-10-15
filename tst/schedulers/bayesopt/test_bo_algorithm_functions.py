from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm \
    import _pick_from_locally_optimized, _lazily_locally_optimize
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components \
    import NoOptimization
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.utils.duplicate_detector \
    import DuplicateDetectorIdentical
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from sagemaker_tune.search_space import uniform, randint, choice
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects \
    import tuples_to_configs, create_exclusion_set

def test_pick_from_locally_optimized():
    duplicate_detector = DuplicateDetectorIdentical()
    hp_ranges = make_hyperparameter_ranges({
        'hp1': uniform(-10.0, 10.0),
        'hp2': uniform(-10.0, 10.0)})
    _original = tuples_to_configs([
        (0.1, 1.0),
        (0.1, 1.0),  # not a duplicate
        (0.2, 1.0),  # duplicate optimized; Resolved by the original
        (0.1, 1.0),  # complete duplicate
        (0.3, 1.0),  # blacklisted original
        (0.4, 3.0),  # blacklisted all
        (1.0, 2.0),  # final candidate to be selected into a batch
        (0.0, 2.0),  # skipped
        (0.0, 2.0),  # skipped
    ], hp_ranges=hp_ranges)
    _optimized = tuples_to_configs([
        (0.1, 1.0),
        (0.6, 1.0),
        (0.1, 1.0),
        (0.1, 1.0),
        (0.1, 1.0),
        (0.3, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.0),
    ], hp_ranges=hp_ranges)
    exclusion_candidates = create_exclusion_set(
        [(0.3, 1.0), (0.4, 3.0), (0.0, 0.0)], hp_ranges)
    got = _pick_from_locally_optimized(
        candidates_with_optimization=list(zip(_original, _optimized)),
        exclusion_candidates=exclusion_candidates,
        num_candidates=4,
        duplicate_detector=duplicate_detector)

    expected = tuples_to_configs(
        [(0.1, 1.0), (0.6, 1.0), (0.2, 1.0), (1.0, 1.0)], hp_ranges=hp_ranges)

    # order of the candidates should be preserved
    assert len(expected) == len(got)
    assert all(a == b for a, b in zip(got, expected))


def test_lazily_locally_optimize():
    hp_ranges = make_hyperparameter_ranges({
        'a': uniform(0.0, 2.0),
        'b': choice(['a', 'c', 'd']),
        'c': randint(0, 3),
        'd': choice(['a', 'b', 'd'])})
    original_candidates = tuples_to_configs(
        [(1.0, 'a', 3, 'b'), (2.0, 'c', 2, 'a'), (0.0, 'd', 0, 'd')], hp_ranges)

    # NoOptimization class is used to check the interfaces only in here
    i = 0
    for candidate in  _lazily_locally_optimize(
            candidates=original_candidates,
            local_optimizer=NoOptimization(None, None, None),
            hp_ranges=hp_ranges,
            model=None):
        # no optimization is applied ot the candidates
        assert candidate[0] == original_candidates[i]
        assert candidate[1] == original_candidates[i]
        i += 1

    assert i == len(original_candidates)
    assert len(list(_lazily_locally_optimize(
        candidates=[],
        local_optimizer=NoOptimization(None, None, None),
        hp_ranges=hp_ranges,
        model=None))) == 0
