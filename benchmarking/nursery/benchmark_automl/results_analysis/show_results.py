# %%
import logging
from argparse import ArgumentParser
from benchmarking.nursery.benchmark_automl.baselines import Methods

from benchmarking.nursery.benchmark_automl.results_analysis.utils import method_styles, load_and_cache, plot_results, \
    print_rank_table

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default="purple-akita",
        help="the experiment tag that was displayed when running the experiment"
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    logging.getLogger().setLevel(logging.INFO)

    load_cache_if_exists = True

    # benchmarks_to_df = {bench: df[] for bench, df in benchmarks_to_df.items()}
    methods_to_show = list(method_styles.keys())
    benchmarks_to_df = load_and_cache(
        load_cache_if_exists=load_cache_if_exists, experiment_tag=experiment_tag, methods_to_show=methods_to_show
    )

    for bench, df_ in benchmarks_to_df.items():
        df_methods = df_.algorithm.unique()
        for x in methods_to_show:
            if x not in df_methods:
                logging.warning(f"method {x} not found in {bench}")

    for benchmark in ['fcnet', 'nas201']:
        n = 0
        for key, df in benchmarks_to_df.items():
            if benchmark in key:
                n += len(df[df.algorithm == Methods.RS])
        print(f"number of hyperband evaluations for {benchmark}: {n}")

    methods_to_show = [
        Methods.RS,
        Methods.TPE,
        Methods.REA,
        Methods.GP,
        Methods.MSR,
        Methods.ASHA,
        Methods.BOHB,
        Methods.MOBSTER,
        Methods.ASHA_BB,
        Methods.ASHA_CTS,
    ]
    print_rank_table(benchmarks_to_df, methods_to_show)

    plot_results(benchmarks_to_df, method_styles)