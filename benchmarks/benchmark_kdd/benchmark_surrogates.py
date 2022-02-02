import pandas as pd
import itertools
import xgboost
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

from blackbox_repository import load
from syne_tune.optimizer.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.transfer_learning.quantile_based.thompson_sampling_functional_prior import \
    extract_input_output, eval_model, fit_model


def benchmark_surrogates():
    # todo 2) roundrobin eval
    # todo 3) compare different surrogates
    blackbox_names = ["fcnet", "nasbench201"]
    rows = []
    surrogates = {
        "xgboost": lambda: xgboost.XGBRegressor(),
        "knn": lambda: KNeighborsRegressor(n_neighbors=3),
    }
    num_samples = [100, 1000, 10000, 100000, 1000000]
    combinations = list(itertools.product(blackbox_names, surrogates.items(), num_samples))

    print(combinations)
    for blackbox_name, (surrogate_name, surrogate), max_fit_samples in tqdm(combinations):
        if surrogate_name == "knn" and max_fit_samples > 10000:
            continue
        print(blackbox_name, surrogate_name, max_fit_samples)
        bb_dict = load(blackbox_name)
        normalization = "gaussian"
        for test_task in bb_dict.keys():
            config_space = bb_dict[test_task].configuration_space
            metric_index = 0
            transfer_learning_evaluations_train = {
                task: TransferLearningTaskEvaluations(
                    hyperparameters=bb.hyperparameters,
                    # average over seed, take last fidelity and pick only first metric
                    metrics=bb.objectives_evaluations.mean(axis=1)[:, -1, metric_index:metric_index + 1]
                )
                for task, bb in bb_dict.items()
                if task != test_task
            }
            transfer_learning_evaluations_test = {
                task: TransferLearningTaskEvaluations(
                    hyperparameters=bb.hyperparameters,
                    # average over seed, take last fidelity and pick only first metric
                    metrics=bb.objectives_evaluations.mean(axis=1)[:, -1, metric_index:metric_index + 1]
                )
                for task, bb in bb_dict.items()
                if task == test_task
            }
            model_pipeline, sigma_train, sigma_val = fit_model(
                config_space=config_space,
                transfer_learning_evaluations=transfer_learning_evaluations_train,
                normalization=normalization,
                max_fit_samples=max_fit_samples,
                model=surrogate(),
            )
            X, y = extract_input_output(transfer_learning_evaluations_test, normalization)
            sigma_test = eval_model(model_pipeline, X, y)
            row = {
                "task": test_task,
                "RMSE-train": sigma_train,
                "RMSE-val": sigma_val,
                "RMSE-test": sigma_test,
                "num_samples": max_fit_samples,
                "surrogate": surrogate_name,
            }
            print(row)
            rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string())
    df.to_csv("surrogates_results.csv", index=False)


if __name__ == '__main__':
    # benchmark_surrogates()
    df = pd.read_csv("surrogates_results.csv")
    print(df.to_string())
    df_pivot = df.pivot_table(index='task', columns=['surrogate', 'num_samples'], values='RMSE-test')
    print(df_pivot.to_string(float_format="%.2f"))