import pandas as pd
import numpy as np


from syne_tune.blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from syne_tune.blackbox_repository.conversion_scripts.blackbox_recipe import (
    BlackboxRecipe,
)
from syne_tune.blackbox_repository.conversion_scripts.scripts import (
    metric_elapsed_time,
    default_metric,
    time_attr,
)
from syne_tune.config_space import randint, uniform, loguniform, choice
from syne_tune.util import catchtime
from syne_tune.blackbox_repository.conversion_scripts.utils import (
    repository_path,
)

from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path

BLACKBOX_NAME_VANILLIX = "autoencodix_vanillix"
BLACKBOX_NAME_VARIX = "autoencodix_varix"

METRIC_DOWNSTREAM_PERFORMANCE = "WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE"
METRIC_ELAPSED_TIME = "training_runtime"
TIME_ATTR = "epoch"

MAX_RESOURCE_LEVEL = 1

CONFIGURATION_SPACE_VANILLIX = {
    # Model Parameters
    "K_FILTER": choice([128, 256, 512, 1024, 2048, 4096]),
    "N_LAYERS": choice([2, 3, 4]),  # 3 values
    "ENC_FACTOR": uniform(1, 4),
    "LATENT_DIM_FIXED": choice([2, 4, 8, 16, 32, 64]),
    # Training Parameters
    "BATCH_SIZE": choice([32, 64, 128, 256]),
    "LR_FIXED": loguniform(1e-5, 1e-1),
    "DROP_P": uniform(0, 0.9),
}

CONFIGURATION_SPACE_VARIX = {
    # Model Parameters
    "K_FILTER": choice([128, 256, 512, 1024, 2048, 4096]),  # 6 values
    "N_LAYERS": choice([2, 3, 4]),  # 3 values
    "ENC_FACTOR": uniform(1, 4),
    "LATENT_DIM_FIXED": choice([2, 4, 8, 16, 32, 64]),  # 6 values
    # Training Parameters
    "BATCH_SIZE": choice([32, 64, 128, 256]),  # 4 values
    "LR_FIXED": loguniform(1e-5, 1e-1),
    "DROP_P": uniform(0, 0.9),
    "BETA": loguniform(0.001, 10),
}


def generate_autoencodix(config_space, blackbox_name):
    with catchtime(f"loading data for {blackbox_name}"):
        # load data frames with results from parquet files
        df_dict = {
            ("tcga", "rna"): pd.read_parquet(
                f"ae_results_30000_runs/real_ae_results_{blackbox_name.split('_')[1]}_tcga_RNA.parquet"
            ),
            ("tcga", "meth"): pd.read_parquet(
                f"ae_results_30000_runs/real_ae_results_{blackbox_name.split('_')[1]}_tcga_METH.parquet"
            ),
            ("tcga", "dna"): pd.read_parquet(
                f"ae_results_30000_runs/real_ae_results_{blackbox_name.split('_')[1]}_tcga_DNA.parquet"
            ),
            ("schc", "rna"): pd.read_parquet(
                f"ae_results_30000_runs/real_ae_results_{blackbox_name.split('_')[1]}_schc_RNA.parquet"
            ),
            ("schc", "meth"): pd.read_parquet(
                f"ae_results_30000_runs/real_ae_results_{blackbox_name.split('_')[1]}_schc_METH.parquet"
            ),
        }

    with catchtime(f"converting {blackbox_name}"):
        hyperparameters_dict = {}
        for (data_scenario, task), df in df_dict.items():
            cols = [
                "K_FILTER",
                "N_LAYERS",
                "ENC_FACTOR",
                "LATENT_DIM_FIXED",
                "LR_FIXED",
                "BATCH_SIZE",
                "DROP_P",
            ]
            hp_df = df[cols].copy()

            if blackbox_name == BLACKBOX_NAME_VARIX:
                hp_df["BETA"] = df["BETA"].values

            hyperparameters_dict[(data_scenario, task)] = hp_df

            # add weighted average, taking into account high correlation between AUC on CANCER_TYPE, SUB_TYPE, ONCOTREE_CODE
            if data_scenario == "tcga":
                df["WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE"] = (
                    df["CANCER_TYPE"] * 1 / 21
                    + df["SUBTYPE"] * 1 / 21
                    + df["ONCOTREE_CODE"] * 1 / 21
                    + df["SEX"] * 1 / 7
                    + df["AJCC_PATHOLOGIC_TUMOR_STAGE"] * 1 / 7
                    + df["GRADE"] * 1 / 7
                    + df["PATH_N_STAGE"] * 1 / 7
                    + df["DSS_STATUS"] * 1 / 7
                    + df["OS_STATUS"] * 1 / 7
                )
            elif data_scenario == "schc":
                df["WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE"] = (
                    df["author_cell_type"] * 1 / 3
                    + df["age_group"] * 1 / 3
                    + df["sex"] * 1 / 3
                )

        objectives = [
            "training_runtime",
            "valid_r2",
            "WEIGHTED_AVG_AUC_DOWNSTREAM_PERFORMANCE",
        ]

        # use a constant value here as no multi-fidelity data is available
        fidelity_space = {time_attr: randint(lower=300, upper=300)}

        bb_dict = {}

        for (data_scenario, task) in df_dict.keys():
            task_name = data_scenario + "_" + task
            # must be a numpy array
            objectives_evaluations = df_dict[(data_scenario, task)][
                objectives
            ].to_numpy()
            # convert (x by y) to (x, 1, 1, y)
            objectives_evaluations = np.expand_dims(
                np.expand_dims(objectives_evaluations, axis=1), axis=2
            )

            bb_dict[task_name] = BlackboxTabular(
                hyperparameters=hyperparameters_dict[(data_scenario, task)],
                configuration_space=config_space,
                fidelity_space=fidelity_space,  # this is a dict
                objectives_evaluations=objectives_evaluations,
                fidelity_values=np.arange(
                    1
                ),  # single [0] in a numpy array for one epoch value
                objectives_names=objectives,
            )

    with catchtime("saving to disk"):
        serialize(
            bb_dict=bb_dict,
            path=repository_path / blackbox_name,
            metadata={
                metric_elapsed_time: METRIC_ELAPSED_TIME,
                default_metric: METRIC_DOWNSTREAM_PERFORMANCE,
                time_attr: TIME_ATTR,
            },
        )


class AutoencodixRecipe(BlackboxRecipe):
    def __init__(self, config_space, name):
        super(AutoencodixRecipe, self).__init__(
            name=name,
            cite_reference="Varix (VAE) from 'A generalized and versatile framework to train and evaluate autoencoders for biological representation learning and beyond: AUTOENCODIX',"
            "Maximilian Joas, Neringa Jurenaite, Dusan Prascevic, Nico Scherf, Jan Ewald. BioRxiv, 2024.",
        )
        self.config_space = config_space

    def _generate_on_disk(self):
        generate_autoencodix(
            self.config_space,
            self.name,
        )


class AutoEncodixVanillixBlackboxRecipe(AutoencodixRecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_VANILLIX, BLACKBOX_NAME_VANILLIX)


class AutoEncodixVarixBlackboxRecipe(AutoencodixRecipe):
    def __init__(self):
        super().__init__(CONFIGURATION_SPACE_VARIX, BLACKBOX_NAME_VARIX)


if __name__ == "__main__":
    import requests
    import zipfile

    ae_data_file = repository_path / "ae_results_30000_runs.zip"
    github_src = (
        "https://github.com/ralf-koenig/ae-st-hpo/raw/main/"
        "ae_results_30000_runs/ae_results_30000_runs.zip"
    )

    if not ae_data_file.exists():
        print(f"did not find {ae_data_file}, downloading {github_src}")

        response = requests.get(github_src)
        response.raise_for_status()

        # Save under repository_path
        with open(ae_data_file, "wb") as f:
            f.write(response.content)

        # Extract ZIP into repository_path
        with zipfile.ZipFile(ae_data_file, "r") as zip_ref:
            zip_ref.extractall(path=repository_path)

    extracted_folder = repository_path / "ae_results_30000_runs"
    recipes = [AutoEncodixVanillixBlackboxRecipe, AutoEncodixVarixBlackboxRecipe]

    for recipe in recipes:
        instance = recipe()
        instance.generate(upload_on_hub=False)
