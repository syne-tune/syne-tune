# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# %%
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from benchmarking.nursery.benchmark_automl.baselines import Methods

from benchmarking.nursery.benchmark_automl.results_analysis.utils import (
    method_styles,
    load_and_cache,
    plot_results,
    print_rank_table,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        required=False,
        default="purple-akita",
        help="the experiment tag that was displayed when running the experiment",
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    logging.getLogger().setLevel(logging.INFO)

    load_cache_if_exists = True

    # benchmarks_to_df = {bench: df[] for bench, df in benchmarks_to_df.items()}
    methods_to_show = list(method_styles.keys())
    benchmarks_to_df = load_and_cache(
        load_cache_if_exists=load_cache_if_exists,
        experiment_tag=experiment_tag,
        methods_to_show=methods_to_show,
    )

    for bench, df_ in benchmarks_to_df.items():
        df_methods = df_.algorithm.unique()
        for x in methods_to_show:
            if x not in df_methods:
                logging.warning(f"method {x} not found in {bench}")

    for benchmark in ["fcnet", "nas201"]:
        n = 0
        for key, df in benchmarks_to_df.items():
            if benchmark in key:
                n += len(df[df.algorithm == Methods.RS])
        print(f"number of hyperband evaluations for {benchmark}: {n}")

    methods_to_show = [
        Methods.RS,
        Methods.TPE,
        Methods.REA,
        # Methods.BORE,
        Methods.GP,
        Methods.MSR,
        Methods.ASHA,
        Methods.BOHB,
        Methods.MOBSTER,
        Methods.RUSH,
        Methods.ASHA_BB,
        Methods.ZERO_SHOT,
        Methods.ASHA_CTS,
    ]
    print_rank_table(benchmarks_to_df, methods_to_show)

    params = {
        "legend.fontsize": 18,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }
    plt.rcParams.update(params)

    plot_results(benchmarks_to_df, method_styles, methods_to_show=methods_to_show)
