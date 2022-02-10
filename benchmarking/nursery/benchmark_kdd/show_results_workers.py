# %%
import logging
import os
from argparse import ArgumentParser

import dill
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from benchmarking.nursery.benchmark_kdd.show_results import generate_df_dict, plot_results, MethodSyle, \
    multifidelity_style, rs_color, load_and_cache
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import get_metadata, load_experiments_df

# %%
from syne_tune.util import catchtime


show_seeds = False


if __name__ == '__main__':
    date_min = datetime.fromisoformat("2022-01-04")
    date_max = datetime.fromisoformat("2023-01-04")

    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default="nworkers-snobbish-toucan",
        help="the experiment tag that was displayed when running the experiment"
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    logging.getLogger().setLevel(logging.INFO)

    cmap = cm.get_cmap("viridis")
    method_styles = {
        'HB (1 workers)': MethodSyle(cmap(0), "-"),
        'HB (2 workers)': MethodSyle(cmap(0.25), "-"),
        'HB (4 workers)': MethodSyle(cmap(0.5), "-"),
        'HB (8 workers)': MethodSyle(cmap(1.0), "-"),
    }

    load_cache_if_exists = False
    methods_to_show = list(method_styles.keys())

    result_file = Path(f"~/Downloads/cached-results-{experiment_tag}.dill").expanduser()

    methods_to_show = list(method_styles.keys())
    benchmarks_to_df = load_and_cache(
        load_cache_if_exists=load_cache_if_exists, experiment_tag=experiment_tag,
        methods_to_show=methods_to_show
    )

    for bench, df_ in benchmarks_to_df.items():
        df_methods = df_.algorithm.unique()
        for x in methods_to_show:
            if x not in df_methods:
                logging.warning(f"method {x} not found in {bench}")

    # benchmarks_to_df = {bench: df[] for bench, df in benchmarks_to_df.items()}

    plot_results(benchmarks_to_df, method_styles, prefix="number-workers-")