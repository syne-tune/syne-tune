from examples.training_scripts.height_example.train_height import (
    height_config_space,
    METRIC_ATTR,
)

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from pathlib import Path
import pytest
import sys


@pytest.mark.skip("too slow")
@pytest.mark.timeout(30)
@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="BoTorch requires python 3.8 or higher"
)
def test_smoke_botorch():
    """
    This test checks that the BoTorch scheduler can be loaded and used to come up with new trials.
    """
    from syne_tune.optimizer.legacy_baselines import BoTorch

    # Use train_height backend for our tests
    entry_point = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    max_trials = 5
    # Set up tuner and run for a few evaluations
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=str(entry_point)),
        scheduler=BoTorch(
            metric=METRIC_ATTR,
            config_space=height_config_space(max_steps=5),
            random_seed=15,
        ),
        stop_criterion=StoppingCriterion(max_num_trials_finished=max_trials),
        n_workers=1,
        sleep_time=0.001,
    )
    tuner.run()

    df = tuner.tuning_status.get_dataframe()

    assert (
        len(df[df["status"] == "Completed"]) >= max_trials
    ), "Checks that we have expected number of trials completed"
