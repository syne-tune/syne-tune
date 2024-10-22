import numpy as np

from syne_tune.optimizer.schedulers.searchers.bayesopt.models.estimator import (
    transform_state_to_data,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    dictionarize_objective,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.test_objects import (
    create_tuning_job_state,
)
from syne_tune.config_space import uniform, randint, choice, loguniform
from syne_tune.optimizer.schedulers.searchers.utils.hp_ranges_factory import (
    make_hyperparameter_ranges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.warping import (
    warpings_for_hyperparameters,
)


def test_get_internal_candidate_evaluations():
    """we do not test the case with no evaluations, as it is assumed
    that there will be always some evaluations generated in the beginning
    of the BO loop."""

    hp_ranges = make_hyperparameter_ranges(
        {"a": randint(0, 10), "b": uniform(0.0, 10.0), "c": choice(["X", "Y"])}
    )
    cand_tuples = [(2, 3.3, "X"), (1, 9.9, "Y"), (7, 6.1, "X")]
    metrics = [dictionarize_objective(y) for y in (5.3, 10.9, 13.1)]

    state = create_tuning_job_state(
        hp_ranges=hp_ranges, cand_tuples=cand_tuples, metrics=metrics
    )
    state.failed_trials.append("0")  # First trial with observation also failed

    result = transform_state_to_data(
        state, INTERNAL_METRIC_NAME, normalize_targets=True, num_fantasy_samples=20
    )

    assert len(result.features.shape) == 2, "Input should be a matrix"
    assert len(result.targets.shape) == 2, "Output should be a matrix"

    assert result.features.shape[0] == len(cand_tuples)
    assert result.targets.shape[-1] == 1, "Only single output value per row is suppored"

    assert (
        np.abs(np.mean(result.targets)) < 1e-8
    ), "Mean of the normalized outputs is not 0.0"
    assert (
        np.abs(np.std(result.targets) - 1.0) < 1e-8
    ), "Std. of the normalized outputs is not 1.0"

    np.testing.assert_almost_equal(result.mean, 9.766666666666666)
    np.testing.assert_almost_equal(result.std, 3.283629428273267)


def test_warpings_for_hyperparameters():
    # Note: ``choice`` with binary value range is encoded as 1, not 2 dims
    hp_ranges = make_hyperparameter_ranges(
        {
            "a": choice(["X", "Y"]),  # pos 0
            "b": loguniform(0.1, 10.0),  # pos 1
            "c": choice(["a", "b", "c"]),  # pos 2
            "d": uniform(0.0, 10.0),  # pos 5
            "e": choice(["X", "Y"]),  # pos 6
        }
    )

    warpings = warpings_for_hyperparameters(hp_ranges)
    assert hp_ranges.ndarray_size == 7
    assert len(warpings) == 2
    assert warpings[0].lower == 1 and warpings[0].upper == 2
    assert warpings[1].lower == 5 and warpings[1].upper == 6
