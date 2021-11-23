"""
Convert tabular data from
NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search
Xuanyi Dong, Yi Yang
https://arxiv.org/abs/2001.00326
"""
import pandas as pd
import numpy as np

import s3fs

import syne_tune.search_space as sp
from blackbox_repository import load
from blackbox_repository.blackbox_tabular import BlackboxTabular, serialize
from blackbox_repository.conversion_scripts.utils import repository_path, upload


def load_nas201_df():
    # todo we start from an processed file for simplicity however,
    # this file only save several metrics kept only a single seed, we should
    # start from the original nas201 dataset instead
    path = repository_path / "nasbench201.csv.zip"
    if not path.exists():
        src = "amper-benchmark-repo/datasets/nas201/nasbench201.csv"
        print(f"did not found {path}, downloading it from {src}")
        s3fs.S3FileSystem().get(src, str(path))
        # compress it
        pd.read_csv(repository_path / "nasbench201.csv").to_csv(repository_path / "nasbench201.csv.zip", index=False)

    return pd.read_csv(repository_path / "nasbench201.csv.zip")


def convert_dataset(df_dataset: pd.DataFrame, dataset: str):
    assert len(df_dataset.dataset.unique()) == 1
    assert df_dataset.dataset.unique()[0] == dataset

    hp_cols = [f"hp_x{i}" for i in range(6)]

    # get hyperparameters with shape (n_eval, n_hp) and add suffix "hp_" to respect the repo convention
    hyperparameters = pd.DataFrame(
        data=df_dataset.loc[:, [col.strip("hp_") for col in hp_cols]].values,
        columns=hp_cols
    )

    configuration_space = {
        node: sp.choice(['avg_pool_3x3', 'nor_conv_3x3', 'skip_connect', 'nor_conv_1x1', 'none'])
        for node in hp_cols
    }

    fidelity_space = {
        "epochs": sp.randint(lower=1, upper=200),
    }

    # df has metrics encoded with columns, for instance: 'lc_valid_epoch_0', 'runtime_epoch_0', 'eval_time_epoch_0
    objectives = []
    for col in ['lc_valid', 'runtime', 'eval_time']:
        cols = [f"{col}_epoch_{i}" for i in range(200)]
        values = df_dataset.loc[df_dataset.dataset == dataset, cols].values
        print(col, values.min(), values.max())
        objectives.append(
            # (n_evals, n_seeds, n_epochs, 1) -> (n_evals, 1, n_epochs, 1) as there is only one seed
            np.expand_dims(values, [1, -1])
        )

    # (n_evals, n_seeds, n_epochs, n_objectives)
    objective_evaluations = np.concatenate(objectives, axis=-1)

    # goes from validation accuracy to error ratio
    objective_evaluations[..., 0] = 1 - objective_evaluations[..., 0] / 100

    fidelity_values = np.arange(1, 201)

    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=configuration_space,
        fidelity_space=fidelity_space,
        objectives_evaluations=objective_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=['metric_error', 'metric_runtime', 'metric_eval_runtime'],
    )


# def plot_learning_curve():
#     import matplotlib.pyplot as plt
#     bb_dict = load("amper-nas201")
#     bb = bb_dict['cifar100']
#     for _ in range(5):
#         configuration = {k: v.sample() for k, v in bb.configuration_space.items()}
#
#         print(configuration)
#         errors = []
#         for i in range(1, 201):
#             res = bb.objective_function(configuration=configuration, fidelity={'epochs': i})
#             errors.append(res['metric_error'])
#         plt.plot(errors)
#     plt.show()
#

def generate_nas201():
    df = load_nas201_df()
    bb_dict = {}
    for dataset in sorted(df.dataset.unique()):
        print(f"converting {dataset}")
        bb_dict[dataset] = convert_dataset(df_dataset=df.loc[df.dataset == dataset, :], dataset=dataset)

    serialize(bb_dict=bb_dict, path=repository_path / "nas201")

    upload("nas201")


if __name__ == '__main__':
    generate_nas201()
    # plot_learning_curve()
