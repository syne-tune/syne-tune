import numpy as np
import pandas as pd

from syne_tune.optimizer.schedulers.transfer_learning.bounding_box import BoundingBox
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.config_space import randint, uniform, choice
from syne_tune.optimizer.schedulers.single_objective_scheduler import SingleObjectiveScheduler


config_space = {
    "x": randint(0, 20),
    "y": uniform(0, 1),
    "z": choice(["a", "b", "c"]),
}

metric = "objective1"
random_seed = 42

def create_task_evaluation():
    num_evals = 100
    num_seeds = 3
    num_fidelity = 1
    num_objectives = 1
    return {"dummy-task-1": TransferLearningTaskEvaluations(
        config_space,
        hyperparameters=pd.DataFrame(
        [
            {
                "x": np.random.randint(0, 10),
                "y": np.random.rand() / 2,
                "z": np.random.choice(["a", "b"]),
            }
            for _ in range(num_evals)
        ]
        ),
        objectives_evaluations=np.arange(num_evals * num_seeds * num_fidelity * num_objectives).reshape(num_evals, num_seeds, num_fidelity, num_objectives),
        objectives_names=[metric])
    }
    
def test_bounding_box():

    data = create_task_evaluation()
    bb = BoundingBox(random_seed=random_seed, config_space=config_space, metric=metric,
                     transfer_learning_evaluations=data,
                     scheduler_fun=lambda new_config_space, metric, do_minimize, random_seed: SingleObjectiveScheduler(
                         new_config_space,
                         searcher="random_search",
                         metric=metric,
                         random_seed=random_seed,
                         do_minimize=do_minimize,
                     ),
                     )
    new_config_space = bb._compute_box(config_space=config_space, transfer_learning_evaluations=data,
                                       num_hyperparameters_per_task=50, mode='min')

    assert new_config_space['x'].lower >= 0
    assert new_config_space['x'].upper <= 10
    assert new_config_space['y'].lower >= 0
    assert new_config_space['y'].upper <=  0.5
    assert new_config_space['z'].categories == ['a', 'b']