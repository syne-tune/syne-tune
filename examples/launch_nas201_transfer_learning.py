from syne_tune.blackbox_repository import load_blackbox, BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)
from syne_tune.optimizer.schedulers.transfer_learning.transfer_learning_task_evaluation import (
    TransferLearningTaskEvaluations,
)
from syne_tune.optimizer.schedulers.transfer_learning.bounding_box import BoundingBox
from syne_tune import StoppingCriterion, Tuner


def load_transfer_learning_evaluations(
    blackbox_name: str, test_task: str, metric: str
) -> dict[str, TransferLearningTaskEvaluations]:
    bb_dict = load_blackbox(blackbox_name)
    metric_index = [
        i
        for i, name in enumerate(bb_dict[test_task].objectives_names)
        if name == metric
    ][0]
    transfer_learning_evaluations = {
        task: TransferLearningTaskEvaluations(
            hyperparameters=bb.hyperparameters,
            configuration_space=bb.configuration_space,
            objectives_evaluations=bb.objectives_evaluations[
                ..., metric_index : metric_index + 1
            ],
            objectives_names=[metric],
        )
        for task, bb in bb_dict.items()
        if task != test_task
    }
    return transfer_learning_evaluations


if __name__ == "__main__":
    blackbox_name = "nasbench201"
    test_task = "cifar100"
    elapsed_time_attr = "metric_elapsed_time"
    metric = "metric_valid_error"
    random_seed = 42
    do_minimize = True
    bb_dict = load_blackbox(blackbox_name)
    transfer_learning_evaluations = load_transfer_learning_evaluations(
        blackbox_name, test_task, metric
    )

    scheduler = BoundingBox(
        scheduler_fun=lambda new_config_space, metric, do_minimize, random_seed: SingleObjectiveScheduler(
            new_config_space,
            do_minimize=do_minimize,
            searcher="random_search",
            metric=metric,
            random_seed=random_seed,
        ),
        config_space=bb_dict[test_task].configuration_space,
        metric=metric,
        num_hyperparameters_per_task=10,
        transfer_learning_evaluations=transfer_learning_evaluations,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=7200)

    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=elapsed_time_attr,
        dataset=test_task,
    )

    # It is important to set ``sleep_time`` to 0 here (mandatory for simulator backend)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=4,
        sleep_time=0,
        # This callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
    )
    tuner.run()

    tuning_experiment = load_experiment(tuner.name)
    print(tuning_experiment)

    print(f"best result found: {tuning_experiment.best_config()}")

    tuning_experiment.plot()
