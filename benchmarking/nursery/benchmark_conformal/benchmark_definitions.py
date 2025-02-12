from dataclasses import dataclass
from typing import Optional, List, Dict


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
    surrogate_kwargs: Optional[Dict] = None
    datasets: Optional[List[str]] = None


# n_full_evals = 100 may be better as 200 evals is very slow for MOBSTER
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


benchmark_definitions = {
    "fcnet-protein": fcnet_benchmark("protein_structure"),
    "fcnet-naval": fcnet_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_benchmark("slice_localization"),
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
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


if __name__ == "__main__":
    from syne_tune.blackbox_repository import load_blackbox

    for benchmark, benchmark_case in benchmark_definitions.items():
        print(benchmark)
        bb = load_blackbox(benchmark_case.blackbox_name)
