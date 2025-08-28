import logging
from pathlib import Path

from syne_tune.backend import LocalBackend
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    random_seed = 31415927
    max_steps = 100
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }
    entry_point = str(
        Path(__file__).parent
        / "training_scripts"
        / "height_example"
        / "train_height.py"
    )

    trial_backend = LocalBackend(entry_point=entry_point)

    # Random search without stopping
    scheduler = RandomSearch(
        config_space,
        do_minimize=True,
        metrics=["mean_loss"],
        random_seed=random_seed,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=20)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        n_workers=n_workers,
        stop_criterion=stop_criterion,
        results_update_interval=5,
        tuner_name="plot-results-demo",
        metadata={"description": "just an example"},
    )

    tuner.run()

    # shows how to print the best configuration found from the tuner and retrains it
    trial_id, best_config = tuner.best_config()

    tuning_experiment = load_experiment(tuner.name)

    # prints the best configuration found from experiment-results
    print(f"best result found: {tuning_experiment.best_config()}")

    # plots the best metric over time
    tuning_experiment.plot()

    # plots values found by all trials over time
    tuning_experiment.plot_trials_over_time()
