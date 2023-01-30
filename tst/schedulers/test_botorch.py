from examples.training_scripts.height_example.train_height import (
    height_config_space,
    METRIC_ATTR,
)

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import BoTorch
import numpy as np
import pytest
import pandas as pd

@pytest.mark.timeout(20)
def test_botorch_reproducible():
    
    # Use train_height backend for our tests
    entry_point = 'examples/training_scripts/height_example/train_height.py'

    # Set up tuner and run for a few evaluations
    tuner = Tuner(
            trial_backend=LocalBackend(entry_point=entry_point),
            scheduler=BoTorch(
                metric=METRIC_ATTR,
                config_space=height_config_space(max_steps=5),
                random_seed=15),
            stop_criterion=StoppingCriterion(max_num_trials_finished=5),
            n_workers=1,
            sleep_time=0.001,
    )
    tuner.run()

    df = tuner.tuning_status.get_dataframe()

    expected = pd.DataFrame({
        'width': [10, 14, 18, 10, 8],
        'height': [0, 68, 23, 59, 100],
        'step': [4, 4, 4, 4, 4],
        'mean_loss': [2.0000, 8.3151, 3.5195, 7.9000, 12.3810]
    })
    
    assert (((expected - df[['width', 'height', 'step', 'mean_loss']][:5]) < 0.0001).all()).all(), 'BoTorch did not choose expected hyperparameters to sample.'