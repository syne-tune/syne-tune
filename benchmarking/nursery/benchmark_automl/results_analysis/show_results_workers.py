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
import logging
from argparse import ArgumentParser
from pathlib import Path
from matplotlib import cm

from benchmarking.nursery.benchmark_automl.baselines import Methods
from benchmarking.nursery.benchmark_automl.results_analysis.utils import (
    MethodStyle,
    load_and_cache,
    plot_results,
)

show_seeds = False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag",
        type=str,
        required=False,
        default="nworkers-rebel-ibex",
        help="the experiment tag that was displayed when running the experiment",
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    logging.getLogger().setLevel(logging.INFO)

    cmap = cm.get_cmap("viridis")
    method_styles = {
        f"{Methods.ASHA} (1 workers)": MethodStyle(cmap(0), "-"),
        f"{Methods.ASHA} (2 workers)": MethodStyle(cmap(0.25), "-"),
        f"{Methods.ASHA} (4 workers)": MethodStyle(cmap(0.5), "-"),
        f"{Methods.ASHA} (8 workers)": MethodStyle(cmap(1.0), "-"),
    }

    load_cache_if_exists = True
    methods_to_show = list(method_styles.keys())

    result_file = Path(f"~/Downloads/cached-results-{experiment_tag}.dill").expanduser()

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

    # benchmarks_to_df = {bench: df[] for bench, df in benchmarks_to_df.items()}

    plot_results(
        benchmarks_to_df,
        method_styles,
        prefix="number-workers-",
        title="Impact of parallelism on wallclock time",
    )
