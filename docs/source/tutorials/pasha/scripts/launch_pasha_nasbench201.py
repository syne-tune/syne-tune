# PASHA: Efficient HPO with Progressive Resource Allocation

from syne_tune.experiments import load_experiment
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.optimizer.baselines import baselines_dict
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from benchmarking.definitions.definition_nasbench201 import (
    nasbench201_benchmark,
    nasbench201_default_params,
)
from benchmarking.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from benchmarking.blackbox_repository import load
import pandas as pd
import logging


def load_nb201(dataset_name, nb201_random_seed):
    """
    Function to load NASBench201 dataframe.

    :param dataset_name: one of 'cifar10', 'cifar100', 'ImageNet16-120'
    :param nb201_random_seed: one of 0, 1, 2
    :return: dataframe with NASBench201 information about the selected dataset
    """

    # Load NASBench201 benchmark
    bb_dict = load("nasbench201")

    metric_valid_error_dim = 0
    metric_runtime_dim = 2
    epoch_names = ["val_acc_epoch_" + str(e) for e in range(200)]

    # create a dataframe with the validation accuracies for various epochs
    df_val_acc = pd.DataFrame(
        (
            1.0
            - bb_dict[dataset_name].objectives_evaluations[
                :, nb201_random_seed, :, metric_valid_error_dim
            ]
        )
        * 100,
        columns=["val_acc_epoch_" + str(e) for e in range(200)],
    )
    # add a new column with the best validation accuracy
    df_val_acc["val_acc_best"] = df_val_acc[epoch_names].max(axis=1)
    # create a dataframe with the hyperparameter values
    df_hp = bb_dict[dataset_name].hyperparameters
    # create a dataframe with the times it takes to run an epoch
    df_time = pd.DataFrame(
        bb_dict[dataset_name].objectives_evaluations[
            :, nb201_random_seed, :, metric_runtime_dim
        ][:, -1],
        columns=["eval_time_epoch"],
    )
    # combine all smaller dataframes into one dataframe
    nb201_df = pd.concat([df_hp, df_val_acc, df_time], axis=1)

    return nb201_df


def run_experiment(
    dataset_name,
    random_seed,
    nb201_random_seed,
    hpo_approach,
    reduction_factor=None,
    rung_system_kwargs={"ranking_criterion": "soft_ranking", "epsilon": 0.025},
):
    """
    Function to run a NASBench201 experiment. It is similar to the NASBench201 example script
    in syne-tune but extended to make it simple to run our experiments.

    :param dataset_name: one of 'cifar10', 'cifar100', 'ImageNet16-120'
    :param random_seed: e.g. 31415927
    :param nb201_random_seed: one of 0, 1, 2
    :param hpo_approach: one of 'pasha', 'pasha-bo'
    :param reduction_factor: by default None (resulting in using the default value 3)
    :param rung_system_kwargs: dictionary of ranking criterion (str) and epsilon or epsilon scaling (both float)
    :return: tuner.name
    """

    logging.getLogger().setLevel(logging.WARNING)

    default_params = nasbench201_default_params({"backend": "simulated"})
    benchmark = nasbench201_benchmark(default_params)
    # benchmark must be tabulated to support simulation
    assert benchmark.get("supports_simulated", False)
    mode = benchmark["mode"]
    metric = benchmark["metric"]
    blackbox_name = benchmark.get("blackbox_name")
    # NASBench201 is a blackbox from the repository
    assert blackbox_name is not None

    config_space = benchmark["config_space"]

    # simulator backend specialized to tabulated blackboxes
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=benchmark["elapsed_time_attr"],
        dataset=dataset_name,
        seed=nb201_random_seed,
    )

    # set logging of the simulator backend to WARNING level
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(
        logging.WARNING
    )

    if not reduction_factor:
        reduction_factor = default_params["reduction_factor"]

    if hpo_approach == "pasha":
        scheduler = baselines_dict["PASHA"](
            config_space,
            max_t=default_params["max_resource_level"],
            grace_period=default_params["grace_period"],
            reduction_factor=reduction_factor,
            resource_attr=benchmark["resource_attr"],
            mode=mode,
            metric=metric,
            random_seed=random_seed,
            rung_system_kwargs=rung_system_kwargs,
        )
    elif hpo_approach == "pasha-bo":
        scheduler = HyperbandScheduler(
            config_space,
            max_t=default_params["max_resource_level"],
            grace_period=default_params["grace_period"],
            reduction_factor=reduction_factor,
            resource_attr=benchmark["resource_attr"],
            mode=mode,
            searcher="bayesopt",
            type="pasha",
            metric=metric,
            random_seed=random_seed,
            rung_system_kwargs=rung_system_kwargs,
        )

    stop_criterion = StoppingCriterion(max_num_trials_started=max_num_trials_started)
    # printing the status during tuning takes a lot of time, and so does
    # storing results
    print_update_interval = 700
    results_update_interval = 300
    # it is important to set ``sleep_time`` to 0 here (mandatory for simulator
    # backend)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        results_update_interval=results_update_interval,
        print_update_interval=print_update_interval,
        # this callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
    )

    tuner.run()

    return tuner.name


def analyse_experiment(experiment_name, nb201_df, reference_time=None):
    """
    Function to analyse the experiments that we run with run_experiment function.

    :param experiment_name: name of the experiment to analyse
    :param nb201_df: dataframe with NASBench201 information about the selected dataset
    :param reference_time: optional argument with the time it takes to run the standard method - e.g. ASHA
    :return: tuple of a line to display (string reporting the experiment results) and
        the mean of the runtimes that can be used as reference time for other approaches
    """

    experiment_results = load_experiment(experiment_name)
    best_cfg = experiment_results.results["metric_valid_error"].argmin()

    # find the best validation accuracy of the corresponding entry in NASBench201
    table_hp_names = ["hp_x" + str(hp_idx) for hp_idx in range(6)]
    results_hp_names = ["config_hp_x" + str(hp_idx) for hp_idx in range(6)]
    condition = (
        nb201_df[table_hp_names]
        == experiment_results.results[results_hp_names].iloc[best_cfg].tolist()
    ).all(axis=1)
    # there is only one item in the list
    val_acc_best = nb201_df[condition]["val_acc_best"].values[0]
    max_rsc = experiment_results.results["hp_epoch"].max()
    runtime = experiment_results.results["st_tuner_time"].max()

    line = " & {:.2f}".format(val_acc_best)
    line += " & {:.1f}h".format(runtime / 3600)
    if reference_time:
        line += " & {:.1f}x".format(reference_time / runtime)
    else:
        line += " & 1.0x"
    line += " & {:.1f}".format(max_rsc)

    return line, runtime


if __name__ == "__main__":
    # Define our settings
    dataset_name = "ImageNet16-120"
    nb201_random_seed = 0
    random_seed = 31415927
    n_workers = 4
    max_num_trials_started = 256
    hpo_approach = "pasha"

    # Initial data loading and pre-processing
    nb201_df = load_nb201(dataset_name, nb201_random_seed)

    # Run an experiment
    experiment_name = run_experiment(
        dataset_name, random_seed, nb201_random_seed, hpo_approach
    )

    # To run PASHA with an alternative ranking function, e.g. soft ranking with $\epsilon=2\sigma$ use
    # rung_system_kwargs={'ranking_criterion': 'soft_ranking_std', 'epsilon_scaling': 2.0}

    # To run an experiment with a different reduction factor, specify the value for ``reduction_factor``

    # To run an experiment with a Bayesian Optimization search strategy, select ``'pasha-bo'`` for ``hpo_approach``

    # Analysis of the results
    print("\nExperiment results:\n")
    result_summary, _ = analyse_experiment(experiment_name, nb201_df)
    print("PASHA" + result_summary)
