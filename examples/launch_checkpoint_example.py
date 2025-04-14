"""
An example showing how to retrieve the best checkpoint of an XGBoost model.
The script being tuned ``xgboost_checkpoint.py`` stores the checkpoint obtained after each trial evaluation.
After the tuning is done, this example loads the best checkpoint and evaluate the model.
"""

import logging
from pathlib import Path

from examples.training_scripts.xgboost.xgboost_checkpoint import evaluate_accuracy
from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.legacy_baselines import BayesianOptimization
from syne_tune import Tuner, StoppingCriterion
import syne_tune.config_space as cs


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    n_workers = 4

    config_space = {
        "max_depth": cs.randint(2, 5),
        "gamma": cs.uniform(1, 9),
        "reg_lambda": cs.loguniform(1e-6, 1),
        "n_estimators": cs.randint(1, 10),
    }

    entry_point = (
        Path(__file__).parent / "training_scripts" / "xgboost" / "xgboost_checkpoint.py"
    )

    trial_backend = LocalBackend(entry_point=str(entry_point))

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=BayesianOptimization(config_space, metric="merror", mode="min"),
        stop_criterion=StoppingCriterion(max_wallclock_time=10),
        n_workers=n_workers,
    )

    tuner.run()

    exp = load_experiment(tuner.name)
    best_config = exp.best_config()
    checkpoint = trial_backend.checkpoint_trial_path(best_config["trial_id"])
    assert checkpoint.exists()

    print(f"Best config found {best_config} checkpointed at {checkpoint}")

    print(
        f"Retrieve best checkpoint and evaluate accuracy of best model: "
        f"found {evaluate_accuracy(checkpoint_dir=checkpoint)}"
    )
