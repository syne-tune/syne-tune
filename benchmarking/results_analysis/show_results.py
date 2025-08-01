import logging
from argparse import ArgumentParser
from pathlib import Path

import dill
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from benchmarking.baselines import (
    Methods,
)
from benchmarking.results_analysis.load_experiments_parallel import (
    load_benchmark_results,
)
from benchmarking.results_analysis.method_styles import (
    plot_range,
)
from syne_tune.util import catchtime


# matplotlib.rcParams["pdf.fonttype"] = 42


def figure_folder(path):
    import syne_tune

    root = Path(syne_tune.__path__[0]).parent
    figure_path = root / path
    figure_path.mkdir(exist_ok=True, parents=True)
    print(figure_path)
    return figure_path


lw = 2.5
alpha = 0.7
matplotlib.rcParams.update({"font.size": 15})
benchmark_families = [
    "fcnet",
    "lcbench",
    "nas201",
    "tabrepo-ExtraTrees",
    "tabrepo-RandomForest",
    "tabrepo-LightGBM",
    "tabrepo-CatBoost",
    # "yahpo"
    "hpob_4796",
    "hpob_5527",
    "hpob_5636",
    "hpob_5859",
    "hpob_5860",
    "hpob_5891",
    "hpob_5906",
    "hpob_5965",
    "hpob_5970",
    "hpob_5971",
    "hpob_6766",
    "hpob_6767",
    "hpob_6794",
    "hpob_7607",
    "hpob_7609",
    "hpob_5889",
]
benchmark_names = {
    "fcnet": "\\FCNet{}",
    "nas201": "\\NASBench{}",
    "lcbench": "\\LCBench{}",
    # "yahpo": "\\NASSurr{}",
}


def plot_result_benchmark(
    t_range: np.array,
    method_dict: dict[str, np.array],
    title: str,
    rename_dict: dict,
    ax=None,
    methods_to_show: list = None,
    plot_regret: bool = True,
):
    agg_results = {}

    if plot_regret:
        mode = "min"
        min_value = min([v.min() for v in method_dict.values()])
        max_value = max([v.max() for v in method_dict.values()])
        best_result, worse_result = (
            (min_value, max_value) if mode == "min" else (max_value, min_value)
        )

    if len(method_dict) > 0:
        if ax is None:
            fig, ax = plt.subplots()
        for algorithm in method_dict.keys():
            if methods_to_show is not None and algorithm not in methods_to_show:
                continue
            renamed_algorithm = rename_dict.get(algorithm, algorithm)

            # (num_seeds, num_time_steps)
            y_ranges = method_dict[algorithm]
            if plot_regret:
                y_ranges = (y_ranges - best_result) / (worse_result - best_result)
            mean = y_ranges.mean(axis=0)
            std = y_ranges.std(axis=0, ddof=1) / np.sqrt(y_ranges.shape[0])
            ax.fill_between(
                t_range,
                mean - std,
                mean + std,
                alpha=0.1,
            )
            ax.plot(
                t_range,
                mean,
                label=renamed_algorithm,
                alpha=alpha,
            )

            agg_results[algorithm] = mean

        ax.set_xlabel("Wallclock time")
        ax.legend()
        ax.set_title(title)
    return ax


def plot_task_performance_over_time(
    benchmark_results: dict[str, tuple[np.array, dict[str, np.array]]],
    rename_dict: dict,
    result_folder: Path,
    title: str = None,
    ax=None,
    methods_to_show: list = None,
    plot_regret: bool = False,
):
    print(f"plot rank through time on {result_folder}")
    for benchmark, (t_range, method_dict) in benchmark_results.items():
        ax = plot_result_benchmark(
            t_range=t_range,
            method_dict=method_dict,
            title=benchmark,
            ax=ax,
            methods_to_show=methods_to_show,
            rename_dict=rename_dict,
            plot_regret=plot_regret,
        )
        ax.set_ylabel("objective")
        if title is not None:
            ax.set_title(title)
        if not plot_regret:
            if benchmark in plot_range:
                plotargs = plot_range[benchmark]
                ax.set_ylim([plotargs.ymin, plotargs.ymax])
                ax.set_xlim([plotargs.xmin, plotargs.xmax])

        if ax is not None:
            plt.tight_layout()
            filepath = result_folder / f"{benchmark}.pdf"
            plt.savefig(filepath)
        ax = None


def load_and_cache(
    path: Path,
    methods: list[str] | None = None,
    load_cache_if_exists: bool = True,
    num_time_steps=100,
    max_seed=10,
    experiment_filter=None,
):
    result_file = (Path(path) / "results-cache.dill").expanduser()
    if load_cache_if_exists and result_file.exists():
        with catchtime(f"loading results from {result_file}"):
            with open(result_file, "rb") as f:
                benchmark_results = dill.load(f)
    else:
        print(f"regenerating results to {result_file}")
        with catchtime("load benchmark results"):
            benchmark_results = load_benchmark_results(
                path=path,
                methods=methods,
                num_time_steps=num_time_steps,
                max_seed=max_seed,
                experiment_filter=experiment_filter,
            )

        with open(result_file, "wb") as f:
            dill.dump(benchmark_results, f)

    return benchmark_results


def plot_ranks(
    ranks,
    benchmark_results,
    title: str,
    rename_dict: dict,
    result_folder: Path,
    methods_to_show: list[str],
):
    plt.figure()
    # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    ys = ranks.reshape(benchmark_results.shape).mean(axis=(1, 2))
    xs = np.linspace(0, 1, ys.shape[-1])
    for i, method in enumerate(methods_to_show):
        plt.plot(
            xs,
            ys[i],
            label=rename_dict.get(method, method),
            alpha=alpha,
            lw=lw,
        )
    plt.xlabel("% Budget Used")
    plt.ylabel("Method rank")
    plt.xlim(0, 1)
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(result_folder / f"{title}.pdf")


def stack_benchmark_results(
    benchmark_results_dict: dict[str, tuple[np.array, dict[str, np.array]]],
    methods_to_show: list[str] | None,
    benchmark_families: list[str],
) -> dict[str, np.array]:
    """
    Stack benchmark results between benchmarks of the same family.
    :param benchmark_results_dict:
    :param methods_to_show:
    :return: dictionary from benchmark family to tensor results with shape
    (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    """
    for benchmark, (t_range, method_dict) in benchmark_results_dict.items():
        for method in methods_to_show:
            if method not in method_dict:
                print(
                    f"removing method {method} from methods to show as it is not present in all benchmarks"
                )
                methods_to_show.remove(method)
    if len(methods_to_show) == 0:
        return {}
    else:
        res = {}
        for benchmark_family in benchmark_families:
            # list of the benchmark of the current family
            benchmarks_family = [
                benchmark
                for benchmark in benchmark_results_dict.keys()
                if benchmark_family in benchmark
            ]

            benchmark_results = []
            for benchmark in benchmarks_family:
                benchmark_result = [
                    benchmark_results_dict[benchmark][1][method]
                    for method in methods_to_show
                ]
                benchmark_result = np.stack(benchmark_result)
                benchmark_results.append(benchmark_result)

            # (num_benchmarks, num_methods, num_min_seeds, num_time_steps)
            benchmark_results = np.stack(benchmark_results)

            if benchmark_family in [
                "lcbench",
                "yahpo",
                "hpob_4796",
                "hpob_5527",
                "hpob_5636",
                "hpob_5859",
                "hpob_5860",
                "hpob_5891",
                "hpob_5906",
                "hpob_5965",
                "hpob_5970",
                "hpob_5971",
                "hpob_6766",
                "hpob_6767",
                "hpob_6794",
                "hpob_7607",
                "hpob_7609",
                "hpob_5889",
            ]:
                # max instead of minimization, todo pass the mode somehow
                benchmark_results *= -1

            # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
            res[benchmark_family] = benchmark_results.swapaxes(0, 1)

        return res


def generate_rank_results(
    benchmark_families: list[str],
    stacked_benchmark_results: dict[str, np.array],
    methods_to_show: list[str] | None,
    rename_dict: dict,
    result_folder: Path,
):
    rows = []
    for benchmark_family in benchmark_families:
        print(benchmark_family)
        # list of the benchmark of the current family
        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        benchmark_results = stacked_benchmark_results[benchmark_family]

        ranks = pd.DataFrame(
            benchmark_results.reshape(len(benchmark_results), -1)
        ).rank()
        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        ranks = ranks.values.reshape(benchmark_results.shape)
        # (num_methods, num_benchmarks)
        avg_ranks_per_tasks = ranks.mean(axis=(2, 3))
        for i in range(benchmark_results.shape[1]):
            row = {"benchmark": f"{benchmark_family}-{i}"}
            row.update(dict(zip(methods_to_show, avg_ranks_per_tasks[:, i])))
            rows.append(row)

        plot_ranks(
            ranks,
            benchmark_results,
            benchmark_family,
            rename_dict,
            result_folder,
            methods_to_show,
        )

    # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    all_results = np.concatenate(list(stacked_benchmark_results.values()), axis=1)
    all_ranks = pd.DataFrame(all_results.reshape(len(all_results), -1)).rank()
    # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    # all_ranks = all_ranks.values.reshape(all_results.shape)
    plot_ranks(
        all_ranks.values,
        all_results,
        "Average-rank",
        rename_dict,
        result_folder,
        methods_to_show,
    )


def plot_average_normalized_regret(
    stacked_benchmark_results,
    rename_dict: dict,
    result_folder: Path,
    title: str = None,
    show_ci: bool = False,
    ax=None,
    methods_to_show: list = None,
):
    normalized_regrets = []
    for benchmark_family in benchmark_families:
        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        benchmark_results = stacked_benchmark_results[benchmark_family]
        # uncomment to remove outliers
        # benchmark_results = np.clip(benchmark_results, a_min=None, a_max=np.percentile(benchmark_results, 99))
        benchmark_results_best = benchmark_results.min(axis=(0, 2, 3), keepdims=True)
        benchmark_results_worse = benchmark_results.max(axis=(0, 2, 3), keepdims=True)
        # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
        normalized_regret = (benchmark_results - benchmark_results_best) / (
            benchmark_results_worse - benchmark_results_best
        )
        normalized_regrets.append(normalized_regret)

    # (num_methods, num_benchmarks, num_min_seeds, num_time_steps)
    normalized_regrets = np.concatenate(normalized_regrets, axis=1)

    # (num_methods, num_time_steps)
    avg_regret = normalized_regrets.mean(axis=(1, 2))
    std_regret = normalized_regrets.std(axis=2).mean(axis=1) if show_ci else None

    if ax is None:
        fig, ax = plt.subplots()
    for i, algorithm in enumerate(methods_to_show):
        renamed_algorithm = rename_dict.get(algorithm, algorithm)
        # (num_seeds, num_time_steps)
        mean = avg_regret[i]
        ax.plot(
            np.arange(len(mean)) / len(mean),
            mean,
            # color=method_style.color,
            # linestyle=method_style.linestyle,
            # marker=method_style.marker,
            label=renamed_algorithm,
            lw=lw,
            alpha=alpha,
        )
        if show_ci:
            std = std_regret[i]
            ax.fill_between(
                np.arange(len(mean)) / len(mean),
                mean - std,
                mean + std,
                # color=method_style.color,
                alpha=0.1,
            )
        ax.set_yscale("log")

    plt.xlabel("% Budget Used")
    ax.set_ylabel("Average normalized regret")
    plt.xlim(0, 1)
    plt.ylim(6e-3, None)
    plt.grid()
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(result_folder / f"{title}.pdf")
    # plt.show()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path where to find the results",
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        required=False,
    )

    methods_selected = [
        Methods.RS,
        Methods.REA,
        Methods.BORE,
        Methods.TPE,
        Methods.CQR,
        Methods.BOTorch,
        Methods.BOHB,
        Methods.ASHA,
        Methods.ASHACQR,
        Methods.ASHABORE,
    ]

    single_fidelity = [x for x in methods_selected if not ("ASHA" in x or "BOHB" in x)]
    multi_fidelity = [x for x in methods_selected if x not in single_fidelity]

    methods_to_show = single_fidelity + multi_fidelity

    groups = {
        "single-fidelity": single_fidelity,
        "multi-fidelity": multi_fidelity,
        "all": single_fidelity + multi_fidelity,
    }

    args, _ = parser.parse_known_args()

    print(args.__dict__)
    assert Path(args.path).exists()
    max_seed = args.max_seed
    num_time_steps = 50

    with catchtime("load benchmark results"):
        benchmark_results = load_and_cache(
            path=args.path,
            load_cache_if_exists=args.reuse_cache,
            max_seed=max_seed,
            num_time_steps=num_time_steps,
            methods=methods_to_show,
        )

    assert (
        len(benchmark_results) > 0
    ), f"Could not find results in path provided {args.path}."
    for group_name, methods in groups.items():
        if len(methods) > 0:
            folder_name = Path(args.path).parent.name
            result_folder = figure_folder(Path("figures") / folder_name / group_name)
            result_folder.mkdir(parents=True, exist_ok=True)
            stacked_benchmark_results = stack_benchmark_results(
                benchmark_results_dict=benchmark_results,
                methods_to_show=methods,
                benchmark_families=benchmark_families,
            )
            if len(stacked_benchmark_results) > 0:
                rename_dict = {}
                with catchtime("generating rank table"):
                    generate_rank_results(
                        stacked_benchmark_results=stacked_benchmark_results,
                        benchmark_families=benchmark_families,
                        methods_to_show=methods,
                        rename_dict=rename_dict,
                        result_folder=result_folder,
                    )

                with catchtime("generating plots per task"):
                    plot_task_performance_over_time(
                        benchmark_results=benchmark_results,
                        methods_to_show=methods,
                        rename_dict=rename_dict,
                        result_folder=result_folder,
                    )

                plot_average_normalized_regret(
                    stacked_benchmark_results=stacked_benchmark_results,
                    methods_to_show=methods,
                    rename_dict=rename_dict,
                    result_folder=result_folder,
                    title="Normalized-regret",
                )
