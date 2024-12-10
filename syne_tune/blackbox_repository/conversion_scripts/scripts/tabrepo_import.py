import pandas as pd
import numpy as np
from tabrepo import load_repository, get_context, list_contexts, EvaluationRepository
from syne_tune.blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.scripts import metric_elapsed_time
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
)
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.config_space import uniform, loguniform, randint, choice
from syne_tune.util import catchtime


BLACKBOX_NAME = "tabrepo_"
RESOURCE_ATTR = "hp_epoch"
MAX_RESOURCE_LEVEL = 100
METRIC_ELAPSED_TIME = "metric_elapsed_time"

# TODO ask about types. How to handle provided defualt value?
# TODO Correct classes used?
# TODO hwo to handle None in choice from LinearModel? Removed for developing purposes
# TODO what about RandomForest, ExtraTrees, heterogenous choice data types as well
CONFIGURATION_SPACE_KNeighbors = {
    "n_neighbors": choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 30, 40, 50]),
    "weights": choice(["uniform", "distance"]),
    "p": choice([2, 1]),
}

CONFIGURATION_SPACE_LinearModel = {
    "C": uniform(lower=0.1, upper=1e3),
    "proc.skew_threshold": choice([0.99, 0.9, 0.999]),
    "proc.impute_strategy": choice(["median", "mean"]),
    "penalty": choice(["L2", "L1"]),
}

CONFIGURATION_SPACE_RandomForest = {
    "max_leaf_nodes": randint(5000, 50000),
    "min_samples_leaf": choice([1, 2, 3, 4, 5, 10, 20, 40, 80]),
    #'max_features': choice(['sqrt', 'log2', 0.5, 0.75, 1.0])
}

CONFIGURATION_SPACE_ExtraTrees = {
    "max_leaf_nodes": randint(5000, 50000),
    "min_samples_leaf": choice([1, 2, 3, 4, 5, 10, 20, 40, 80]),
    #'max_features': choice(['sqrt', 'log2', 0.5, 0.75, 1.0])
}

CONFIGURATION_SPACE_MLP = {
    "learning_rate": loguniform(1e-4, 3e-2),
    "weight_decay": loguniform(1e-12, 0.1),
    "dropout_prob": uniform(0.0, 0.4),
    "use_batchnorm": choice([False, True]),
    "num_layers": randint(1, 5),
    "hidden_size": randint(8, 256),
    "activation": choice(["relu", "elu"]),
}

CONFIGURATION_SPACE_XGBoost = {
    "learning_rate": loguniform(lower=5e-3, upper=0.1),
    "max_depth": randint(lower=4, upper=10),
    "min_child_weight": uniform(0.5, 1.5),
    "colsample_bytree": uniform(0.5, 1.0),
    "enable_categorical": choice([True, False]),
}

CONFIGURATION_SPACE_LightGBM = {
    "learning_rate": loguniform(
        lower=5e-3,
        upper=0.1,
    ),
    "feature_fraction": uniform(lower=0.4, upper=1.0),
    "min_data_in_leaf": randint(lower=2, upper=60),
    "num_leaves": randint(lower=16, upper=255),
    "extra_trees": choice([False, True]),
}

CONFIGURATION_SPACE_CatBoost = {
    "learning_rate": loguniform(lower=5e-3, upper=0.1),
    "depth": uniform(lower=4, upper=8),
    "l2_leaf_reg": uniform(
        lower=1,
        upper=5,
    ),
    "max_ctr_complexity": uniform(lower=1, upper=5),
    "one_hot_max_size": choice([2, 3, 5, 10]),
    "grow_policy": choice(["SymmetricTree", "Depthwise"]),
}


def generate_tabrepo(config_space):
    print("generating")

    blackbox_name = BLACKBOX_NAME
    bb_dict = {}

    # D244_F3_C1530 for full datasets
    context_name = "D244_F3_C1530_30"

    repo: EvaluationRepository = load_repository(context_name, cache=True)

    config = "CatBoost_r1_BAG_L1"
    config_type = repo.config_type(config=config)
    config_hyperparameters = repo.config_hyperparameters(config=config)

    metrics = repo.metrics(datasets=repo.datasets(), configs=repo.configs())

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    ):
        print(f"Config Metrics Example:\n{metrics}")

    for dataset in repo.datasets():
        bb_dict[dataset] = convert_dataset(metrics)
    with catchtime("saving to disk"):
        serialize(
            bb_dict=bb_dict,
            path=repository_path / blackbox_name,
            metadata={metric_elapsed_time: METRIC_ELAPSED_TIME},
        )


def convert_dataset(config_space):
    # names of hyperparameters
    hp_cols = list(config_space.keys())
    # number of hyperparameters
    n_hps = len(hp_cols)
    # number of evaluations
    n_evals = None
    # initialize hyperparameter array with shape(n_evals, n_hps)
    hps = np.zeros((n_evals, n_hps))

    for i, config in enumerate(dataset["X"]):
        hps[i] = config

    hyperparameters = pd.DataFrame(data=hps, columns=hp_cols)

    objective_names = [
        "metric_error",
        "metric_error_val",
        "metric_time_train_s",
        "metric_time_infer_s",
        "metric_rank",
    ]

    objective_evaluations = np.array(dataset["y"])  # np.array.shape = (N,)
    objective_evaluations = objective_evaluations.reshape(
        objective_evaluations.shape[0], 1, 1, 1
    )

    # Create a metric_elapsed_time array filled with ones as runtime was not provided in the dataset
    elapsed_time_array = np.ones_like(objective_evaluations)

    objective_evaluations = np.concatenate(
        [objective_evaluations, elapsed_time_array], axis=-1
    )
    # fidelity space initialized as constant value, since it is required as an argument
    fidelity_space = {
        RESOURCE_ATTR: randint(lower=MAX_RESOURCE_LEVEL, upper=MAX_RESOURCE_LEVEL)
    }

    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=config_space,
        objectives_evaluations=objective_evaluations,
        objectives_names=objective_names,
        fidelity_space=fidelity_space,
    )


class TABREPORecipe(BlackboxRecipe):
    def __init__(self, config_space, name):
        super(TABREPORecipe, self).__init__(
            name=name,
            cite_reference="TabRepo: A Large Scale Repository of Tabular Model Evaluations and its Auto{ML} Applications"
            "David Salinas and Nick Erickson",
        )
        self.config_space = config_space

        def _generate_on_disk(self):
            generate_tabrepo(self.config_space)


class TABREPORecipeKNeighbors(TABREPORecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_KNeighbors, BLACKBOX_NAME + "KNeighbors")


class TABREPOMLP(TABREPORecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_MLP, BLACKBOX_NAME + "MLP")


class TABREPOExtraTrees(TABREPORecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_ExtraTrees, BLACKBOX_NAME + "ExtraTrees")


class TABREPOCatBoost(TABREPORecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_CatBoost, BLACKBOX_NAME + "CatBoost")


class TABREPOXGBoost(TABREPORecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_XGBoost, BLACKBOX_NAME + "XGBoost")


class TABREPOLightGBM(TABREPORecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_LightGBM, BLACKBOX_NAME + "LightGBM")


class TABREPOLinearModel(TABREPORecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_LinearModel, BLACKBOX_NAME + "LinearModel")


class TABREPORandomForest(TABREPORecipe):
    def __init__(self):
        super().__init__(
            CONFIGURATION_SPACE_RandomForest, BLACKBOX_NAME + "RandomForest"
        )


if __name__ == "__main__":
    """recipes = [TABREPORecipeKNeighbors,
               TABREPOMLP,
               TABREPOExtraTrees,
               TABREPOCatBoost,
               TABREPOXGBoost,
               TABREPOLightGBM,
               TABREPOLinearModel,
               TABREPORandomForest
               ]

    for recipe in recipes:
        instance = recipe()
        instance.generate(upload_on_hub=False)"""
