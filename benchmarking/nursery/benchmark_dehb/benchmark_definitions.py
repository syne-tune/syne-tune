from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BenchmarkDefinition:
    max_wallclock_time: float
    n_workers: int
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    max_resource_attr: str
    time_this_resource_attr: Optional[str] = None
    max_num_evaluations: Optional[int] = None
    surrogate: Optional[str] = None
    datasets: Optional[List[str]] = None


def fcnet_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=1200,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_loss",
        mode="min",
        blackbox_name="fcnet",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
    )


NAS201_MAX_WALLCLOCK_TIME = {
    "cifar10": 5 * 3600,
    "cifar100": 6 * 3600,
    "ImageNet16-120": 8 * 3600,
}


NAS201_N_WORKERS = {
    "cifar10": 4,
    "cifar100": 4,
    "ImageNet16-120": 8,
}


def nas201_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=NAS201_MAX_WALLCLOCK_TIME[dataset_name],
        n_workers=NAS201_N_WORKERS[dataset_name],
        elapsed_time_attr="metric_elapsed_time",
        time_this_resource_attr="metric_runtime",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
    )


def lcbench_benchmark(dataset_name, datasets):
    return BenchmarkDefinition(
        max_wallclock_time=7200,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        max_num_evaluations=4000,
        datasets=datasets,
        max_resource_attr="epochs",
    )


benchmark_definitions = {
    "fcnet-protein": fcnet_benchmark("protein_structure"),
    "fcnet-naval": fcnet_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_benchmark("slice_localization"),
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
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
