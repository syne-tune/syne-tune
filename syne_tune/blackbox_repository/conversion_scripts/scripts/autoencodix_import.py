"""
Convert experiment data from AutoEncodix hyperparameter optimization runs into Syne Tune blackboxes.
Experiments were done by Ralf König and all the code and data are publicly available at:
https://github.com/ralf-koenig/ae-st-hpo/tree/main
"""
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
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path

BLACKBOX_NAME_VANILLIX = "autoencodix_vanillix"
BLACKBOX_NAME_VARIX = "autoencodix_varix"

METRIC_DOWNSTREAM_PERFORMANCE = "weighted_avg_auc_downstream_performance"
METRIC_ELAPSED_TIME = "training_runtime"
TIME_ATTR = "epoch"

MAX_RESOURCE_LEVEL = 1

CONFIGURATION_SPACE_VANILLIX = {
    # Model Parameters
    "k_filter": choice([128, 256, 512, 1024, 2048, 4096]),
    "n_layers": choice([2, 3, 4]),  # 3 values
    "enc_factor": uniform(1, 4),
    "latent_dim_fixed": choice([2, 4, 8, 16, 32, 64]),
    # Training Parameters
    "batch_size": choice([32, 64, 128, 256]),
    "lr_fixed": loguniform(1e-5, 1e-1),
    "drop_p": uniform(0, 0.9),
}

CONFIGURATION_SPACE_VARIX = {
    # Model Parameters
    "k_filter": choice([128, 256, 512, 1024, 2048, 4096]),  # 6 values
    "n_layers": choice([2, 3, 4]),  # 3 values
    "enc_factor": uniform(1, 4),
    "latent_dim_fixed": choice([2, 4, 8, 16, 32, 64]),  # 6 values
    # Training Parameters
    "batch_size": choice([32, 64, 128, 256]),  # 4 values
    "lr_fixed": loguniform(1e-5, 1e-1),
    "drop_p": uniform(0, 0.9),
    "beta": loguniform(0.001, 10),
}


def generate_autoencodix(config_space, blackbox_name):
    with catchtime(f"loading data for {blackbox_name}"):
        # load data frames with results from parquet files
        architecture = blackbox_name.split("_")[1]
        df_dict = {
            ("tcga", "rna"): pd.read_parquet(
                f"{repository_path}/real_ae_results_{architecture}_tcga_RNA.parquet"
            ),
            ("tcga", "meth"): pd.read_parquet(
                f"{repository_path}/real_ae_results_{architecture}_tcga_METH.parquet"
            ),
            ("tcga", "dna"): pd.read_parquet(
                f"{repository_path}/real_ae_results_{architecture}_tcga_DNA.parquet"
            ),
            ("schc", "rna"): pd.read_parquet(
                f"{repository_path}/real_ae_results_{architecture}_schc_RNA.parquet"
            ),
            ("schc", "meth"): pd.read_parquet(
                f"{repository_path}/real_ae_results_{architecture}_schc_METH.parquet"
            ),
        }

    with catchtime(f"converting {blackbox_name}"):
        hyperparameters_dict = {}
        for (data_scenario, task), df in df_dict.items():
            df.columns = df.columns.str.lower()
            cols = [
                "k_filter",
                "n_layers",
                "enc_factor",
                "latent_dim_fixed",
                "lr_fixed",
                "batch_size",
                "drop_p",
            ]

            hp_df = df[cols].copy()

            if blackbox_name == BLACKBOX_NAME_VARIX:
                hp_df["beta"] = df["beta"].values

            hyperparameters_dict[(data_scenario, task)] = hp_df

            # add weighted average, taking into account high correlation between AUC on CANCER_TYPE, SUB_TYPE, ONCOTREE_CODE
            if data_scenario == "tcga":
                df["weighted_avg_auc_downstream_performance"] = (
                    df["cancer_type"] * 1 / 21
                    + df["subtype"] * 1 / 21
                    + df["oncotree_code"] * 1 / 21
                    + df["sex"] * 1 / 7
                    + df["ajcc_pathologic_tumor_stage"] * 1 / 7
                    + df["grade"] * 1 / 7
                    + df["path_n_stage"] * 1 / 7
                    + df["dss_status"] * 1 / 7
                    + df["os_status"] * 1 / 7
                )
            elif data_scenario == "schc":
                df["weighted_avg_auc_downstream_performance"] = (
                    df["author_cell_type"] * 1 / 3
                    + df["age_group"] * 1 / 3
                    + df["sex"] * 1 / 3
                )

        objectives = [
            "training_runtime",
            "valid_r2",
            "weighted_avg_auc_downstream_performance",
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


def download_autoencodix_data_if_necessary():
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

    download_autoencodix_data_if_necessary()

    recipes = [AutoEncodixVanillixBlackboxRecipe, AutoEncodixVarixBlackboxRecipe]

    for recipe in recipes:
        instance = recipe()
        instance.generate(upload_on_hub=True)
