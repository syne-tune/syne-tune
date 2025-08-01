import json
import logging
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path

import numpy as np
import pandas as pd
from pyparfor import parfor
from tqdm import tqdm

from syne_tune.constants import ST_TUNER_TIME
from syne_tune.util import catchtime


def load_result(name, metadata, path):
    usecols = [metadata["metric_names"][0], "st_tuner_time", "trial_id", "st_decision"]
    try:
        return pd.read_csv(path / name / "results.csv.zip", usecols=usecols)
    except Exception:
        return None


def convert_to_numpy(benchmark_df, num_time_steps: int = 20):
    t_min = 0
    # the last time step is the median of the stopping time of all algorithms
    t_max = benchmark_df.groupby(["algorithm"]).max().st_tuner_time.median()
    t_range = np.linspace(t_min, t_max, num_time_steps)
    seed_results = {}
    for algorithm in benchmark_df["algorithm"].unique():
        ts = []
        ys = []
        df_scheduler = benchmark_df[benchmark_df["algorithm"] == algorithm]
        for seed in sorted(df_scheduler["seed"].unique()):
            sub_df = df_scheduler[df_scheduler["seed"] == seed]
            ts.append(sub_df.loc[:, ST_TUNER_TIME].values)
            ys.append(sub_df.loc[:, "best"].values)

        # for each seed, find the best value at each regularly spaced time-step
        y_ranges = []
        for t, y in zip(ts, ys):
            indices = np.searchsorted(t, t_range, side="left")
            y_range = y[np.clip(indices, 0, len(y) - 1)]
            y_ranges.append(y_range)

        # (num_seeds, num_time_steps)
        y_ranges = np.stack(y_ranges)
        seed_results[algorithm] = y_ranges
    return t_range, seed_results


def compute_best(dfs, metadatas):
    benchmark_dfs = defaultdict(list)
    for df_bench, metadata in tqdm(list(zip(dfs, metadatas.values()))):
        if df_bench is None:
            continue
        benchmark = metadata["benchmark"]
        metric_name = metadata["metric_names"][0]
        for key in ["algorithm", "seed"]:
            df_bench[key] = metadata[key]
        # TODO this avoids the need to rely on the mode stored in the metadata but hardcode the benchmark mode,
        #  we should pass it instead
        if "lcbench" in benchmark or "nas301" in benchmark:
            mode = "max"
        else:
            mode = "min"
        if mode == "min":
            df_bench["best"] = df_bench[metric_name].cummin()
        else:
            df_bench["best"] = df_bench[metric_name].cummax()
            # todo perhaps rename column in O(1) for convenience
        benchmark_dfs[benchmark].append(df_bench)

    benchmark_dfs = {
        k: pd.concat(v, ignore_index=True) for k, v in benchmark_dfs.items()
    }
    return benchmark_dfs


def show_number_seeds(benchmark_dfs):
    df_seeds = pd.DataFrame()
    for benchmark, df in benchmark_dfs.items():
        df_seeds[benchmark] = (
            benchmark_dfs[benchmark]
            .drop_duplicates(["algorithm", "seed"])
            .groupby(["algorithm"])
            .count()["trial_id"]
        )
    print("number of seeds available:")
    print(df_seeds.to_string())


def convert_all_to_numpy(
    benchmark_dfs: dict[str, pd.DataFrame],
    num_time_steps: int,
    max_seed: int,
    engine: str,
):
    benchmarks_numpy = parfor(
        f=convert_to_numpy,
        inputs=[
            {"benchmark_df": benchmark_df, "num_time_steps": num_time_steps}
            for benchmark_df in benchmark_dfs.values()
        ],
        engine=engine,
    )
    min_num_seeds = min(
        values.shape[0] for x in benchmarks_numpy for method, values in x[1].items()
    )
    if max_seed is not None and min_num_seeds < max_seed:
        logging.warning(
            f"some methods have only {min_num_seeds} instead of the {max_seed} seeds asked, slicing all results "
            f"to {min_num_seeds} seeds."
        )
        num_seeds = {
            method: values.shape[0]
            for x in benchmarks_numpy
            for method, values in x[1].items()
        }
        logging.warning(num_seeds)
        benchmarks_numpy = [
            (
                t_range,
                {
                    method: values[:min_num_seeds]
                    for method, values in method_values.items()
                },
            )
            for (t_range, method_values) in benchmarks_numpy
        ]
    return dict(zip(benchmark_dfs.keys(), benchmarks_numpy))


def get_metadata(root: Path):
    metadatas = {}
    for metadata_path in root.rglob(f"*metadata.json"):
        with open(metadata_path, "r") as f:
            folder = metadata_path.parent.name
            try:
                metadatas[folder] = json.load(f)
            except JSONDecodeError as e:
                print(metadata_path)
                raise e

    return metadatas


def load_benchmark_results(
    path: str | Path,
    methods: list[str],
    num_time_steps: int = 20,
    max_seed: int = None,
    experiment_filter=None,
    engine: str = "joblib",
) -> dict[str, tuple[np.array, dict[str, np.array]]]:
    """
    :param path: where results are stored
    :param methods: list of methods to consider
    :param num_time_steps: number of time steps considered for results aggregation
    :param max_seed: maximum seed to load, default to None to load all seeds
    :param experiment_filter:
    :param engine: parallel engine to use, can be ["sequential", "ray", "joblib", "futures"]
    :return:
    """
    path = Path(path)

    with catchtime("Load metadata"):
        metadatas = get_metadata(root=path)

    # todo strict metadata filtering as the one above may fail
    methods = set(methods) if methods is not None else None
    metadatas = {
        k: v
        for k, v in metadatas.items()
        if (max_seed is None or v["seed"] < max_seed)
        and (methods is None or v["algorithm"] in methods)
    }
    if experiment_filter:
        metadatas = {k: v for k, v in metadatas.items() if experiment_filter(v)}
    print(f"loaded {len(metadatas)} experiment metadata")
    # metadatas = {k: v for k, v in metadatas.items() if "yahpo" not in v["benchmark"]}

    with catchtime("Load results dataframes"):
        # load results in parallel

        dfs = parfor(
            lambda name, metadata: load_result(name, metadata, path),
            inputs=list(metadatas.items()),
            engine=engine,
        )
    with catchtime("Compute best result over time"):
        benchmark_dfs = compute_best(dfs, metadatas)

    show_number_seeds(benchmark_dfs)

    with catchtime("Convert to numpy (num_seeds, num_time_steps)"):
        benchmark_results = convert_all_to_numpy(
            benchmark_dfs,
            num_time_steps,
            max_seed,
            engine=engine,
        )
    return benchmark_results
