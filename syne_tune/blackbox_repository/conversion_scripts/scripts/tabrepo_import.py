import pandas as pd
import numpy as np
from tabrepo import load_repository, EvaluationRepository
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

# D244_F3_C1530 for full datasets
CONTEXT_NAME = "D244_F3_C1530_3"

CONFIGURATION_SPACE_KNeighbors = {
    "n_neighbors": choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 30, 40, 50]),
    "weights": choice(["uniform", "distance"]),
    "p": choice([2, 1]),
}

CONFIGURATION_SPACE_LinearModel = {
    "C": uniform(lower=0.1, upper=1e3),
    "proc.skew_threshold": choice(["0.99", "0.9", "0.999", "None"]),
    "proc.impute_strategy": choice(["median", "mean"]),
    "penalty": choice(["L2", "L1"]),
}

CONFIGURATION_SPACE_RandomForest = {
    "max_leaf_nodes": randint(5000, 50000),
    "min_samples_leaf": choice([1, 2, 3, 4, 5, 10, 20, 40, 80]),
    "max_features": choice(["sqrt", "log2", "0.5", "0.75", "1.0"]),
}

CONFIGURATION_SPACE_ExtraTrees = {
    "max_leaf_nodes": randint(5000, 50000),
    "min_samples_leaf": choice([1, 2, 3, 4, 5, 10, 20, 40, 80]),
    "max_features": choice(["sqrt", "log2", "0.5", "0.75", "1.0"]),
}

CONFIGURATION_SPACE_NeuralNetTorch = {
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


def generate_tabrepo(config_space: dict, bb_name: str, context_name: str):
    print(f"generating {bb_name}")

    bb_dict = {}

    repo: EvaluationRepository = load_repository(
        context_name, cache=True, load_predictions=False
    )

    # We collect metrics for all frameworks from tabrepo
    metrics = repo.metrics(datasets=repo.datasets(), configs=repo.configs())
    # We select the metrics for the current blackbox method
    metrics = metrics[
        metrics.index.get_level_values("framework").str.split("_").str[0]
        == bb_name.split("_")[1]
    ]
    # We exclude configurations containing _c suffix
    # The _c configs are configurations that were default configurations of AG at the time of TabRepo
    # and have now been replaced by portfolio _r configurations
    metrics = metrics[metrics.index.get_level_values("framework").str.contains("_c") == False]
    # We iterate over each dataset and pass the metrics for the corresponding hyperparameter configurations and
    # search space to the convert_dataset() function
    for dataset_name, group in metrics.groupby("dataset"):
        print(f"Processing dataset: {dataset_name}")
        hyperparameters_configurations = {}
        for framework in group.index.to_frame(index=False)["framework"].values:
            hyperparameters_configurations[framework] = repo.config_hyperparameters(
            config=framework
            )

        bb_dict[dataset_name] = convert_dataset(
            config_space, group, hyperparameters_configurations
        )

    with catchtime("saving to disk"):
        serialize(
            bb_dict=bb_dict,
            path=repository_path / bb_name,
            metadata={metric_elapsed_time: METRIC_ELAPSED_TIME},
        )


def convert_dataset(
    config_space: dict, evaluations: pd.DataFrame, configurations: dict
):
    print(f"type is: {type(evaluations)})")
    # names of hyperparameters
    hp_cols = list(config_space.keys())
    # number of hyperparameters
    n_hps = len(hp_cols)
    # number of evaluations
    n_evals = len(evaluations.xs(0, level="fold"))
    # initialize hyperparameter array with shape(n_evals, n_hps)
    hps = np.zeros((n_evals, n_hps), dtype=object)

    n_seeds = 3

    for i, config in enumerate(configurations):
        arr = []
        for key in config_space.keys():
            # this avoids type errors in fastparquet conversion, applies only to RandomForrest and ExtraTrees
            if key == "max_features":
                arr.append(str(configurations[config][key]))
            else:
                arr.append(configurations[config][key])
        hps[i] = arr

    hyperparameters = pd.DataFrame(data=hps, columns=hp_cols)
    objective_names = [
        "metric_error",
        "metric_error_val",
        "metric_elapsed_time",
        "metric_time_infer_s",
        "metric_rank",
    ]
    n_objectives = len(objective_names)

    objective_evaluations = []

    for i, fold in enumerate([0, 1, 2]):
        fold_data = evaluations.xs(fold, level="fold")
        for j, (_, row) in enumerate(fold_data.iterrows()):
            metrics = row[
                [
                    "metric_error",
                    "metric_error_val",
                    "time_train_s",
                    "time_infer_s",
                    "rank",
                ]
            ].values

            if len(objective_evaluations) <= j:
                objective_evaluations.append([[], [], []])

            objective_evaluations[j][i] = metrics
    objective_evaluations = np.array(objective_evaluations)
    objective_evaluations = objective_evaluations.reshape(
        n_evals, n_seeds, 1, n_objectives
    )
    # Tabrepo does not provide performance across different fidelities.
    # We initialize the fidelity space as a constant value, since it is required as an argument
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


class TabrepoRecipe(BlackboxRecipe):
    def __init__(self, config_space, name):
        super(TabrepoRecipe, self).__init__(
            name=name,
            cite_reference="TabRepo: A Large Scale Repository of Tabular Model Evaluations and its Auto{ML} Applications"
            "David Salinas and Nick Erickson"
            "AutoML Conference 2024 (ABCD Track)",
        )
        self.config_space = config_space

    def _generate_on_disk(self):
        generate_tabrepo(self.config_space, self.name, CONTEXT_NAME)


class TabrepoRecipeKNeighbors(TabrepoRecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_KNeighbors, BLACKBOX_NAME + "KNeighbors")


class TabrepoNeuralNetTorch(TabrepoRecipe):
    def __init__(self):
        super().__init__(
            CONFIGURATION_SPACE_NeuralNetTorch, BLACKBOX_NAME + "NeuralNetTorch"
        )


class TabrepoExtraTrees(TabrepoRecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_ExtraTrees, BLACKBOX_NAME + "ExtraTrees")


class TabrepoCatBoost(TabrepoRecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_CatBoost, BLACKBOX_NAME + "CatBoost")


class TabrepoXGBoost(TabrepoRecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_XGBoost, BLACKBOX_NAME + "XGBoost")


class TabrepoLightGBM(TabrepoRecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_LightGBM, BLACKBOX_NAME + "LightGBM")


class TabrepoLinearModel(TabrepoRecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_LinearModel, BLACKBOX_NAME + "LinearModel")


class TabrepoRandomForest(TabrepoRecipe):
    def __init__(self):
        super().__init__(
            CONFIGURATION_SPACE_RandomForest, BLACKBOX_NAME + "RandomForest"
        )


if __name__ == "__main__":
    recipes = [
        TabrepoRecipeKNeighbors,
        TabrepoNeuralNetTorch,
        TabrepoRandomForest,
        TabrepoExtraTrees,
        TabrepoCatBoost,
        TabrepoXGBoost,
        TabrepoLightGBM,
        # TabrepoLinearModel
    ]

    for recipe in recipes:
        instance = recipe()
        instance.generate(upload_on_hub=False)
