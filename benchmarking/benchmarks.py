from dataclasses import dataclass
from typing import Optional

from syne_tune.blackbox_repository.conversion_scripts.scripts.tabrepo_import import (
    TABREPO_DATASETS,
)


@dataclass
class BenchmarkDefinition:
    max_wallclock_time: float
    n_workers: int
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    max_num_evaluations: Optional[int] = None
    surrogate: Optional[str] = None
    surrogate_kwargs: Optional[dict] = None
    datasets: Optional[list[str]] = None


n_full_evals = 200


def fcnet_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=7200,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_loss",
        mode="min",
        blackbox_name="fcnet",
        dataset_name=dataset_name,
        # allow to stop after having seen the equivalent of `n_full_evals` evaluations
        max_num_evaluations=100 * n_full_evals,
    )


def nas201_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=72000 if dataset_name == "ImageNet16-120" else 36000,
        max_num_evaluations=200 * n_full_evals,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
    )


def lcbench_benchmark(dataset_name, datasets):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=52 * n_full_evals,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
        datasets=datasets,
    )


def tabrepo_benchmark(blackbox_name: str, dataset_name: str, datasets: list[str]):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=1 * n_full_evals,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",  # todo should also include time_train_s + time_infer_s as metric
        metric="metric_error_val",  # could also do rank
        mode="min",
        blackbox_name=blackbox_name,
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
        datasets=datasets,
    )


benchmark_definitions = {
    "fcnet-protein": fcnet_benchmark("protein_structure"),
    "fcnet-naval": fcnet_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_benchmark("slice_localization"),
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
    # TODO currently fails
    # "nas301-yahpo": BenchmarkDefinition(
    #     max_wallclock_time=3600 * 100,
    #     elapsed_time_attr="runtime",
    #     metric="val_accuracy",
    #     blackbox_name="yahpo-nb301",
    #     dataset_name="CIFAR10",
    #     mode="max",
    #     n_workers=4,
    #     max_num_evaluations=97 * n_full_evals,
    # ),
}


# 5 most expensive lcbench datasets
lc_bench_datasets = [
    "Fashion-MNIST",
    "airlines",
    "albert",
    "covertype",
    "christine",
]
for task in lc_bench_datasets:
    benchmark_definitions[
        "lcbench-" + task.replace("_", "-").replace(".", "")
    ] = lcbench_benchmark(task, datasets=lc_bench_datasets)


# We select a sublist of search spaces
tabrepo_search_spaces = [
    "RandomForest",
    "CatBoost",
    "LightGBM",
    "NeuralNetTorch",
    "ExtraTrees",
]
# TODO find list of 10 representative datasets among the 200+ available
tabrepo_datasets = TABREPO_DATASETS[:10]
for task in tabrepo_datasets:
    for search_space in tabrepo_search_spaces:
        benchmark_definitions[
            f"tabrepo-{search_space}-" + task.replace("_", "-").replace(".", "")
        ] = tabrepo_benchmark(
            blackbox_name=f"tabrepo_{search_space}",
            dataset_name=task,
            datasets=tabrepo_datasets,
        )

if __name__ == "__main__":
    from syne_tune.blackbox_repository import load_blackbox

    for benchmark, benchmark_case in benchmark_definitions.items():
        print(benchmark)
        bb = load_blackbox(benchmark_case.blackbox_name)
