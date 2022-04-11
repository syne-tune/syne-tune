# PASHA: Efficient HPO with Progressive Resource Allocation

# Hyperparameter optimization(HPO) and neural architecture search(NAS)
# are methods of choice to obtain the best-in-class machine learning models,
# but in practice they can be costly to run. When models are trained
# on large datasets, tuning them with HPO or NAS rapidly becomes
# prohibitively expensive for practitioners, even when efficient methods
# such as multi-fidelity ones are employed. To decrease the cost,
# practitioners adopt ad-hoc heuristics with mixed results. We propose
# an approach to tackle the challenge of tuning machine learning models
# trained on large datasets with limited computational resources.
# Our approach, named PASHA, is able to dynamically allocate resources
# for the tuning procedure depending on the need. The experimental comparison
# shows that PASHA identifies well-performing hyperparameter configurations
# and architectures while consuming e.g. 2.8 times fewer computational
# resources than solutions like ASHA.


from syne_tune.experiments import load_experiment
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.optimizer.baselines import baselines_dict
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from benchmarking.definitions.definition_nasbench201 import nasbench201_benchmark, nasbench201_default_params
from syne_tune.blackbox_repository.simulated_tabular_backend import BlackboxRepositoryBackend
from syne_tune.blackbox_repository import load
import random
import pandas as pd
import numpy as np
import logging


def run_experiment(dataset_name, random_seed, nb201_random_seed, hpo_approach,
                   reduction_factor=None, rung_system_kwargs={'ranking_criterion': 'soft_ranking', 'epsilon': 0.025}):
    """
    Function to run a NASBench201 experiment. It is similar to the NASBench201 example script
    in syne-tune but extended to make it simple to run our experiments.

    When describing the following parameters we say what values we use, but feel free to also use other values.

    :param dataset_name: one of 'cifar10', 'cifar100', 'ImageNet16-120'
    :param random_seed: one of 31415927, 0, 1234, 3458, 7685
    :param nb201_random_seed: one of 0, 1, 2
    :param hpo_approach: one of 'pasha', 'asha', 'pasha-bo', 'asha-bo'
    :param reduction_factor: by default None (resulting in using the default value 3) or 2, 4
    :param rung_system_kwargs: dictionary of ranking criterion (str) and epsilon or epsilon scaling (both float)
    :return: tuner.name
    """

    # this function is similar to the NASBench201 example script
    logging.getLogger().setLevel(logging.WARNING)

    default_params = nasbench201_default_params({'backend': 'simulated'})
    benchmark = nasbench201_benchmark(default_params)
    # benchmark must be tabulated to support simulation
    assert benchmark.get('supports_simulated', False)
    mode = benchmark['mode']
    metric = benchmark['metric']
    blackbox_name = benchmark.get('blackbox_name')
    # NASBench201 is a blackbox from the repository
    assert blackbox_name is not None

    config_space = benchmark['config_space']

    # simulator back-end specialized to tabulated blackboxes
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=benchmark['elapsed_time_attr'],
        time_this_resource_attr=benchmark.get('time_this_resource_attr'),
        dataset=dataset_name,
        seed=nb201_random_seed)

    # set logging of the simulator backend to WARNING level
    logging.getLogger(
        'syne_tune.backend.simulator_backend.simulator_backend').setLevel(logging.WARNING)

    if not reduction_factor:
        reduction_factor = default_params['reduction_factor']

    # we support various schedulers within the function
    if hpo_approach == 'pasha':
        scheduler = baselines_dict['PASHA'](
            config_space,
            max_t=default_params['max_resource_level'],
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=benchmark['resource_attr'],
            mode=mode,
            metric=metric,
            random_seed=random_seed,
            rung_system_kwargs=rung_system_kwargs)
    elif hpo_approach == 'asha':
        scheduler = baselines_dict['ASHA'](
            config_space,
            max_t=default_params['max_resource_level'],
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=benchmark['resource_attr'],
            mode=mode,
            type='promotion',
            metric=metric,
            random_seed=random_seed)
    elif hpo_approach == 'pasha-bo':
        scheduler = HyperbandScheduler(
            config_space,
            max_t=default_params['max_resource_level'],
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=benchmark['resource_attr'],
            mode=mode,
            searcher='bayesopt',
            type='pasha',
            metric=metric,
            random_seed=random_seed,
            rung_system_kwargs=rung_system_kwargs)
    elif hpo_approach == 'asha-bo':
        scheduler = HyperbandScheduler(
            config_space,
            max_t=default_params['max_resource_level'],
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=benchmark['resource_attr'],
            mode=mode,
            searcher='bayesopt',
            type='promotion',
            metric=metric,
            random_seed=random_seed)
    else:
        raise ValueError('The selected scheduler is not implemented')

    stop_criterion = StoppingCriterion(max_num_trials_started=256)
    # printing the status during tuning takes a lot of time, and so does
    # storing results
    print_update_interval = 700
    results_update_interval = 300
    # it is important to set `sleep_time` to 0 here (mandatory for simulator
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


def analyse_experiments(experiment_names_dict, reference_time=None):
    """
    Function to analyse the experiments that we run with run_experiment function.

    :param experiment_names_dict: dictionary mapping the dataset names to tuples of
        experiment names and NASBench201 random seeds
    :reference_time: optional argument with the time it takes to run the standard method - e.g. ASHA
    :return: tuple of a line to display (string reporting the experiment results) and 
        the mean of the runtimes that can be used as reference time for other approaches
    """

    val_acc_best_list = []
    max_rsc_list = []
    runtime_list = []

    for experiment_name, nb201_random_seed in experiment_names_dict[dataset_name]:
        experiment_results = load_experiment(experiment_name)
        best_cfg = experiment_results.results['metric_valid_error'].argmin()

        # find the best validation accuracy of the corresponding entry in NASBench201
        table_hp_names = ['hp_x' + str(hp_idx) for hp_idx in range(6)]
        results_hp_names = ['config_hp_x' + str(hp_idx) for hp_idx in range(6)]
        condition = (df_dict[nb201_random_seed][dataset_name][table_hp_names] ==
                     experiment_results.results[results_hp_names].iloc[best_cfg].tolist()).all(axis=1)
        # there is only one item in the list
        val_acc_best = df_dict[nb201_random_seed][dataset_name][condition]['val_acc_best'].values[0]
        val_acc_best_list.append(val_acc_best)
        max_rsc_list.append(experiment_results.results['hp_epoch'].max())
        runtime_list.append(experiment_results.results['st_tuner_time'].max())

    line = ' & {:.2f} $\pm$ {:.2f}'.format(
        np.mean(val_acc_best_list), np.std(val_acc_best_list))
    line += ' & {:.1f}h $\pm$ {:.1f}h'.format(
        np.mean(runtime_list)/3600, np.std(runtime_list)/3600)
    if reference_time:
        line += ' & {:.1f}x'.format(reference_time/np.mean(runtime_list))
    else:
        line += ' & {:.1f}x'.format(np.mean(runtime_list) /
                                    np.mean(runtime_list))
    line += ' & {:.1f} $\pm$ {:.1f}'.format(
        np.mean(max_rsc_list), np.std(max_rsc_list))

    return line, np.mean(runtime_list)


def compute_one_epoch_baseline():
    """
    Function to compute the performance of a simple one epoch baseline.
    :return: a line to display (string reporting the experiment results)
    """

    best_val_obj_list = []
    total_time_list = []

    for nb201_random_seed in nb201_random_seeds:
        for random_seed in random_seeds:
            # randomly sample 256 configurations for the given dataset and NASBench201 seed
            # use the same seeds as for our other experiments
            random.seed(random_seed)
            cfg_list = random.sample(
                range(len(df_dict[nb201_random_seed][dataset_name])), 256)
            selected_subset = df_dict[nb201_random_seed][dataset_name].iloc[cfg_list]
            # find configuration with the best performance after doing one epoch
            max_idx = selected_subset['val_acc_epoch_0'].argmax()
            best_configuration = selected_subset.iloc[max_idx]
            # find the best validation accuracy of the selected configuration
            # as that is the metric that we compare
            best_val_obj = best_configuration[epoch_names].max()

            # we also need to calculate the time it took for this
            # taking into account the number of workers
            total_time = selected_subset['eval_time_epoch'].sum() / n_workers

            best_val_obj_list.append(best_val_obj)
            total_time_list.append(total_time)

    line = ' & {:.2f} $\pm$ {:.2f}'.format(
        np.mean(best_val_obj_list), np.std(best_val_obj_list))
    line += ' & {:.1f}h $\pm$ {:.1f}h'.format(
        np.mean(total_time_list)/3600, np.std(total_time_list)/3600)
    line += ' & {:.1f}x'.format(reference_time/np.mean(total_time_list))
    line += ' & 1.0 $\pm$ 0.0'

    return line


def compute_random_baseline():
    """
    Function to compute the performance of a simple random configuration baseline.

    We consider a ten times larger number of configurations in this case to get a better
    estimate of the performance of a random configuration.

    :return: a line to display (string reporting the experiment results)
    """

    random.seed(0)
    random_seeds_rb = random.sample(range(999999), 256 * 10)

    best_val_obj_list = []
    total_time_list = []

    for nb201_random_seed in nb201_random_seeds:
        for random_seed in random_seeds_rb:
            random.seed(random_seed)
            # select the random configurations
            cfg_list = random.sample(
                range(len(df_dict[nb201_random_seed][dataset_name])), 1)
            selected_configuration = df_dict[nb201_random_seed][dataset_name].iloc[cfg_list]
            # find the best validation accuracy of the selected configuration
            # as that is the metric that we compare
            best_val_obj = selected_configuration[epoch_names].max()

            # we also need to calculate the time it took for this
            total_time = 0.0

            best_val_obj_list.append(best_val_obj)
            total_time_list.append(total_time)

    line = ' & {:.2f} $\pm$ {:.2f}'.format(
        np.mean(best_val_obj_list), np.std(best_val_obj_list))
    line += ' & {:.1f}h $\pm$ {:.1f}h'.format(
        np.mean(total_time_list)/3600, np.std(total_time_list)/3600)
    line += ' & NA'
    line += ' & 0.0 $\pm$ 0.0'

    return line


if __name__ == '__main__':
    # Outline:
    # * Initial pre-processing
    # * Main experiments on NASBench201- with PASHA, ASHA and the baselines
    # * Alternative ranking functions
    # * Changes to the reduction factor
    # * Combination with Bayesian Optimization
    # * Analysis of the results

    # Define our settings
    metric_valid_error_dim = 0
    metric_runtime_dim = 2
    dataset_names = ['cifar10', 'cifar100', 'ImageNet16-120']
    epoch_names = ['val_acc_epoch_' + str(e) for e in range(200)]
    random_seeds = [31415927, 0, 1234, 3458, 7685]
    nb201_random_seeds = [0, 1, 2]
    n_workers = 4


    # Initial pre-processing:

    # Load NASBench201 benchmark so that we can analyse the performance of various approaches
    bb_dict = load('nasbench201')
    df_dict = {}

    for seed in nb201_random_seeds:
        df_dict[seed] = {}
        for dataset in dataset_names:
            # create a dataframe with the validation accuracies for various epochs
            df_val_acc = pd.DataFrame((1.0-bb_dict[dataset].objectives_evaluations[:, seed, :, metric_valid_error_dim]) * 100,
                                      columns=['val_acc_epoch_' + str(e) for e in range(200)])
            # add a new column with the best validation accuracy
            df_val_acc['val_acc_best'] = df_val_acc[epoch_names].max(axis=1)
            # create a dataframe with the hyperparameter values
            df_hp = bb_dict[dataset].hyperparameters
            # create a dataframe with the times it takes to run an epoch
            df_time = pd.DataFrame(bb_dict[dataset].objectives_evaluations[:, seed, :, metric_runtime_dim][:, -1],
                                   columns=['eval_time_epoch'])
            # combine all smaller dataframes into one dataframe for each NASBench201 random seed and dataset
            df_dict[seed][dataset] = pd.concat(
                [df_hp, df_val_acc, df_time], axis=1)

    # Motivation to measure best validation accuracy: NASBench201 provides validation
    # and test errors in an inconsistent format and in fact we can only get the errors
    # for each epoch on their combined validation and test sets for CIFAR-100 and ImageNet16-120.
    # As a tradeoff, we use the combined validation and test sets as the validation set.
    # Consequently, there is no test set which we can use for additional evaluation
    # and so we use the best validation accuracy as the final evaluation metric.


    # Main experiments:
    # We perform experiments on NASBench201 - CIFAR-10, CIFAR-100 and ImageNet16-120 datasets.
    # We use PASHA, ASHA (promotion type) and the relevant baselines - one epoch and random.

    experiment_names_pasha = {dataset: [] for dataset in dataset_names}
    experiment_names_asha = {dataset: [] for dataset in dataset_names}
    for dataset_name in dataset_names:
        for nb201_random_seed in nb201_random_seeds:
            for random_seed in random_seeds:
                experiment_name = run_experiment(
                    dataset_name, random_seed, nb201_random_seed, 'pasha')
                experiment_names_pasha[dataset_name].append(
                    (experiment_name, nb201_random_seed))
                experiment_name = run_experiment(
                    dataset_name, random_seed, nb201_random_seed, 'asha')
                experiment_names_asha[dataset_name].append(
                    (experiment_name, nb201_random_seed))


    # Alternative ranking functions:
    # We show how to run experiments using an alternative ranking function,
    # more specifically soft ranking with $\epsilon=2\sigma$.

    experiment_names_pasha_std2 = {dataset: [] for dataset in dataset_names}

    for dataset_name in dataset_names:
        for nb201_random_seed in nb201_random_seeds:
            for random_seed in random_seeds:
                experiment_name = run_experiment(dataset_name, random_seed,
                                                 nb201_random_seed, 'pasha', rung_system_kwargs={
                                                     'ranking_criterion': 'soft_ranking_std', 'epsilon_scaling': 2.0})
                experiment_names_pasha_std2[dataset_name].append(
                    (experiment_name, nb201_random_seed))


    # Changes to the reduction factor:
    # To run experiments with a different reduction factor,
    # it is enough to specify the value for `reduction_factor`
    # argument provided to `run_experiment` function.


    # Combination with Bayesian Optimization:
    # To run experiments with a Bayesian Optimization search strategy,
    # you need to select `'pasha-bo'` or `'asha-bo'` for `hpo_approach`
    # argument provided to `run_experiment` function. Note these experiments
    # take longer to run because Gaussian processes are used.


    # Analysis of the results:

    print('\nMain experiments:\n')
    for dataset_name in dataset_names:
        print(dataset_name)
        result_summary, reference_time = analyse_experiments(
            experiment_names_asha)
        print('ASHA' + result_summary)
        result_summary, _ = analyse_experiments(
            experiment_names_pasha, reference_time)
        print('PASHA' + result_summary)
        result_summary = compute_one_epoch_baseline()
        print('One epoch baseline', result_summary)
        result_summary = compute_random_baseline()
        print('Random baseline', result_summary)

    # The results show PASHA obtains a similar accuracy as ASHA,
    # but it can find a well-performing configuration much faster.

    # The configurations found by one epoch baseline and random baseline
    # usually obtain significantly lower accuracies, making them unsuitable
    # for finding well-performing configurations.

    print('\nExperiments with alternative ranking functions:\n')
    for dataset_name in dataset_names:
        print(dataset_name)
        result_summary, reference_time = analyse_experiments(
            experiment_names_asha)
        print('ASHA' + result_summary)

        result_summary, _ = analyse_experiments(
            experiment_names_pasha, reference_time)
        print('PASHA soft ranking $\epsilon=0.025$' + result_summary)

        result_summary, _ = analyse_experiments(
            experiment_names_pasha_std2, reference_time)
        print('PASHA soft ranking $2\sigma$' + result_summary)
