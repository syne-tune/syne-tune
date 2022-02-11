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

from benchmarking.nursery.benchmark_kdd.results_analysis.utils import method_styles, load_and_cache, plot_results, \
    print_rank_table
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments import get_metadata, load_experiments_df

# %%
from syne_tune.util import catchtime

if __name__ == '__main__':
    date_min = datetime.fromisoformat("2022-01-04")
    date_max = datetime.fromisoformat("2023-01-04")

    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default="efficient-seagull",
        help="the experiment tag that was displayed when running the experiment"
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    logging.getLogger().setLevel(logging.INFO)

    load_cache_if_exists = False

    # benchmarks_to_df = {bench: df[] for bench, df in benchmarks_to_df.items()}
    methods_to_show = list(method_styles.keys())
    benchmarks_to_df = load_and_cache(load_cache_if_exists=load_cache_if_exists, experiment_tag=experiment_tag, methods_to_show=methods_to_show)
    for bench, df_ in benchmarks_to_df.items():
        df_methods = df_.algorithm.unique()
        for x in methods_to_show:
            if x not in df_methods:
                logging.warning(f"method {x} not found in {bench}")

    plot_results(benchmarks_to_df, method_styles)

    print_rank_table(benchmarks_to_df, methods_to_show)
