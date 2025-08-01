from syne_tune.config_space import randint

import shutil
from pathlib import Path

from syne_tune import StoppingCriterion
from syne_tune import Tuner
from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import ASHA
from syne_tune.util import random_string


def launch_first_tuning(experiment_name: str):
    max_epochs = 100
    metric = "mean_loss"
    do_minimize = True
    config_space = {
        "steps": max_epochs,
        "width": randint(0, 10),
        "height": randint(0, 10),
    }

    entry_point = (
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    scheduler = ASHA(
        config_space=config_space,
        metric=metric,
        do_minimize=do_minimize,
        max_t=max_epochs,
        time_attr="epoch",
    )

    trial_backend = LocalBackend(entry_point=str(entry_point))

    stop_criterion = StoppingCriterion(
        max_num_trials_started=10,
    )
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=4,
        tuner_name=experiment_name,
        suffix_tuner_name=False,
    )

    tuner.run()


if __name__ == "__main__":
    experiment_name = f"resume-tuning-example-{random_string(5)}"

    # Launch a tuning, tuning results and checkpoints are written to disk
    print("Launch first tuning")
    launch_first_tuning(experiment_name)

    print("First tuning done")
    # Later loads an experiment from disk given the experiment name,
    # in particular sets `load_tuner` to True to deserialize the Tuner
    tuning_experiment = load_experiment(experiment_name, load_tuner=True)

    # Optional: copy the tuner as it will be modified when retuning
    # shutil.copy(
    #     tuning_experiment.path / "tuner.dill",
    #     tuning_experiment.path / "tuner-backup.dill",
    # )

    # Update stop criterion to run the tuning a couple more trials than before
    tuning_experiment.tuner.stop_criterion = StoppingCriterion(
        max_num_trials_started=20
    )

    # Resume the tuning
    print(f"Loaded first tuning from disk with following:\n{tuning_experiment}")
    print("Resume tuning")
    tuning_experiment.tuner.run()

    # reload with new info from disk
    tuning_experiment = load_experiment(experiment_name, load_tuner=True)

    print(f"Second tuning finished:\n{tuning_experiment}")
