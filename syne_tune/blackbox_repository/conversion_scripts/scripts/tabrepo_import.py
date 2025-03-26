# To re-generate this benchmark you need to install tabrepo from https://github.com/autogluon/tabrepo.git
# The import currently resides in the generate tabrepo function.
import numpy as np
import pandas as pd
from syne_tune.blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.scripts import (
    metric_elapsed_time,
)
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
)
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.config_space import (
    uniform,
    choice,
    loguniform,
    randint,
)
from syne_tune.util import catchtime

BLACKBOX_NAME = "tabrepo_"
RESOURCE_ATTR = "hp_epoch"
MAX_RESOURCE_LEVEL = 100
METRIC_ELAPSED_TIME = "metric_elapsed_time"

# D244_F3_C1530 for full datasets
CONTEXT_NAME = "D244_F3_C1530"

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
    from tabrepo import load_repository, EvaluationRepository

    print(f"generating {bb_name}")

    bb_dict = {}
    repo: EvaluationRepository = load_repository(
        context_name, cache=True, load_predictions=False
    )
    default_metrics = repo.metrics(
        datasets=repo.datasets(), configs=["ExtraTrees_c1_BAG_L1"]
    )
    # We collect metrics for all frameworks from tabrepo
    metrics = repo.metrics(datasets=repo.datasets(), configs=repo.configs())
    # Choose the desired method from bb_name and filter out _c configurations.
    desired_method = bb_name.split("_")[1]
    filtered_metrics = metrics[
        metrics.index.get_level_values("framework").str.split("_").str[0]
        == desired_method
    ]
    filtered_metrics = filtered_metrics[
        ~filtered_metrics.index.get_level_values("framework").str.contains("_c")
    ]

    # Precompute configurations for every unique framework.
    unique_frameworks = filtered_metrics.index.get_level_values("framework").unique()
    all_configurations = repo.configs_hyperparameters(configs=unique_frameworks)

    # Process each dataset.
    for dataset_name, group in filtered_metrics.groupby("dataset"):
        # For these datasets baseline results are missing, so we do not add them to avoid NaN values.
        if dataset_name in ["KDDCup99", "dionis", "Kuzushiji-49"]:
            continue
        # For the dataset, get the configurations for the frameworks used.
        bb_dict[dataset_name] = convert_dataset(
            config_space, group, all_configurations, default_metrics, dataset_name
        )
    with catchtime("saving to disk"):
        serialize(
            bb_dict=bb_dict,
            path=repository_path / bb_name,
            metadata={metric_elapsed_time: METRIC_ELAPSED_TIME},
        )


def convert_dataset(
    config_space: dict,
    evaluations: pd.DataFrame,
    all_configurations: dict,
    default_metrics: pd.DataFrame,
    dataset_name: str,
):
    # Storing configuration keys to build a consistent ordering of configurations
    all_config_keys = list(all_configurations.keys())
    hp_cols = list(config_space.keys())
    n_hps = len(hp_cols)
    n_evals = len(all_config_keys)

    # Create hyperparameters array using the complete configurations.
    # This avoids to rewrite the serialize() function,
    # as BlackBoxTabular allows only same-sized Blackboxes
    # Missing Evaluations are set to the metrics of "ExtraTrees_c1_BAG_L1" config on the same dataset and fold
    # if it does not exist, it is set to np.nan
    hps = np.empty((n_evals, n_hps), dtype=object)
    for i, config in enumerate(all_config_keys):
        config_dict = all_configurations[config]
        arr = []
        for key in hp_cols:
            # Convert to string for key that causes type issues
            if key == "max_features":
                arr.append(str(config_dict[key]))
            else:
                arr.append(config_dict[key])
        hps[i] = arr
    hyperparameters = pd.DataFrame(data=hps, columns=hp_cols)

    # Objective evaluations provided by TabRepo.
    objective_names = [
        "metric_error",
        "metric_error_val",
        "time_train_s",
        "time_infer_s",
        "rank",
    ]
    n_objectives = len(objective_names)
    n_seeds = (
        3  # We use seeds as folds here, corresponding to folds 0, 1, and 2 from TabRepo
    )

    # Initialize evaluations array with np.nan for missing evaluations
    objective_evaluations = np.full(
        (n_evals, n_seeds, 1, n_objectives), np.nan, dtype=float
    )

    # Loop over each fold (seed) and insert the default evaluation
    for seed_idx, fold in enumerate(range(n_seeds)):
        try:
            fold_data = evaluations.xs(fold, level="fold")
        except KeyError:
            continue

        # Iterate over each row in the fold
        for idx, row in fold_data.iterrows():
            # Get config_id, i.e. the current framework.
            config_id = idx[-1]
            if config_id in all_config_keys:
                pos = all_config_keys.index(config_id)
                # fills all metrics with the performance of the ExtraTrees_c1_BAG_L1 model
                try:
                    # fill the framework evaluation with the default performance of the "ExtraTrees_c1_BAG_L1" model
                    objective_evaluations[pos, seed_idx, 0, :] = default_metrics.loc[
                        (dataset_name, fold, "ExtraTrees_c1_BAG_L1")
                    ].values.astype(float)
                except KeyError:
                    print(
                        f"Got KeyError for {dataset_name}/{fold}, using np.nan instead"
                    )
                    continue

    # Loop over each fold (seed) and insert the actual model evaluations
    for seed_idx, fold in enumerate([0, 1, 2]):
        try:
            fold_data = evaluations.xs(fold, level="fold")
        except KeyError:
            continue

        # Iterate over each row in the fold
        for idx, row in fold_data.iterrows():
            # Get config_id, i.e. the current framework.
            config_id = idx[-1]
            if config_id in all_config_keys:
                pos = all_config_keys.index(config_id)
                metrics = row[objective_names].values.astype(float)
                # matches the right framework with the right position in the objective evaluations array
                objective_evaluations[pos, seed_idx, 0, :] = metrics

    # rename time_train_s to metric_elapsed_time, as this is the default naming for SyneTune
    objective_names[2] = METRIC_ELAPSED_TIME

    # TabRepo does not have multi-fidelity evaluations. We initialize the fidelity space as a constant.
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
        TabrepoLinearModel,
    ]

    for recipe in recipes:
        instance = recipe()
        instance.generate(upload_on_hub=False)
