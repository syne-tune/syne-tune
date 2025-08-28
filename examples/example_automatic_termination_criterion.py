import matplotlib.pyplot as plt

from syne_tune.stopping_criterions.automatic_termination_criterion import (
    AutomaticTerminationCriterion,
)
from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import BORE
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
from syne_tune.blackbox_repository import load_blackbox


n_workers = 4
blackbox_name, dataset, metric = "nasbench201", "cifar100", "metric_valid_error"
elapsed_time_attr = "metric_elapsed_time"
mode = "min"
blackbox = load_blackbox(blackbox_name)[dataset]
max_wallclock_time = 3600 * 6
num_seeds = 10

criterions = ["time", "auto"]
fig, axis = plt.subplots(1, 2, dpi=200, figsize=(12, 6), sharex=True, sharey=True)

for i, criterion in enumerate(criterions):
    axis[i].set_title(criterion)

    for seed in range(num_seeds):
        trial_backend = BlackboxRepositoryBackend(
            elapsed_time_attr=elapsed_time_attr,
            blackbox_name=blackbox_name,
            dataset=dataset,
            seed=seed % 3,
        )

        blackbox = trial_backend.blackbox

        if criterion == "time":
            stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

        else:
            stop_criterion = AutomaticTerminationCriterion(
                blackbox.configuration_space,
                threshold=0.1,
                metric=metric,
                mode=mode,
                seed=seed,
            )

        scheduler = BORE(
            config_space=blackbox.configuration_space,
            metric=metric,
            random_seed=seed,
        )

        print_update_interval = 700
        results_update_interval = 300
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
            sleep_time=0,
            results_update_interval=results_update_interval,
            print_update_interval=print_update_interval,
            callbacks=[SimulatorCallback()],
        )
        tuner.run()

        exp = load_experiment(tuner.name)
        traj = list(exp.results[metric].cummin())
        runtime = list(exp.results["st_tuner_time"])

        axis[i].plot(runtime, traj, color=f"C{i}")
        axis[i].plot(runtime[-1], traj[-1], color=f"C{i}", marker="o")

        axis[i].set_ylim(0.05, 0.3)
        axis[i].set_xlim(0, max_wallclock_time)
fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.ylabel("validation error")
plt.xlabel("wall-clock time (seconds)")
plt.show()
