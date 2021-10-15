"""
Example showing how to tune multiple objectives at once of an artificial function.
We then show the distribution of the hyperparameters sampled to illustrate the difference between different
multiobjective strategies and how linear scalarization fails to provide optimal results.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sagemaker_tune.backend.local_backend import LocalBackend
from sagemaker_tune.experiments import load_experiment
from sagemaker_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from sagemaker_tune.optimizer.schedulers.multiobjective.multiobjective_priority import NonDominatedPriority
from sagemaker_tune.tuner import Tuner
from sagemaker_tune.search_space import uniform
from sagemaker_tune.stopping_criterion import StoppingCriterion


def load_df_results(name):
    dfs = []
    for f in Path("./").glob(f"{name}*.csv"):
        print(f)
        dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)

    max_steps = 27
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "theta": uniform(0, np.pi / 2),
        "sleep_time": 0.01,
    }
    entry_point = Path(__file__).parent / "training_scripts" / "mo_artificial" / "mo_artificial.py"
    mode = "min"

    # we run only nd sort for speed, and display the hyperparameters sampled which provides a better coverage of the
    # pareto from than other strategies.
    priorities = {
        "ndsort": NonDominatedPriority(),
        # "linear-scalarization": LinearScalarizationPriority(),
        # "single-objective": FixedObjectivePriority(),
    }
    num_seeds = 2
    tuner_names = {}
    for priority_name, priority in priorities.items():
        for seed in range(num_seeds):
            np.random.seed(seed)
            scheduler = MOASHA(
                max_t=max_steps,
                time_attr="step",
                mode=mode,
                metrics=["y1", "y2"],
                config_space=config_space,
                multiobjective_priority=priority
            )
            # Local back-end
            backend = LocalBackend(entry_point=str(entry_point))

            stop_criterion = StoppingCriterion(max_wallclock_time=30)
            tuner = Tuner(
                backend=backend,
                scheduler=scheduler,
                stop_criterion=stop_criterion,
                n_workers=n_workers,
                sleep_time=0.5,
            )
            tuner.run()
            tuner_names[priority_name] = tuner.name

    for priority_name in priorities.keys():
        df = load_experiment(tuner_names[priority_name]).results
        df[df.step >= df.step.max()].config_theta.hist().plot()
        plt.title(priority_name)
        plt.show()
