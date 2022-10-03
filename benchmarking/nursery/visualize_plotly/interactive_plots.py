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
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
from IPython import get_ipython
from IPython.display import clear_output, display
from syne_tune.constants import SYNE_TUNE_DEFAULT_FOLDER
from syne_tune.experiments import load_experiment


def plot_tuner_results(
    tuner_name: str,
    experiment_name: str,
    overwrite: bool = False,
    return_df: bool = False,
    refresh: bool = False,
    refresh_rate: int = 60,
    height: Optional[int] = None,
    width: Optional[int] = None,
    max_rows: int = 100,
    max_cols: int = 100,
    display_dataframe: bool = True,
    display_metadata: bool = False,
    dimensions_max_cardinality: int = 50,
    save_figure: bool = False,
) -> Optional[pd.DataFrame]:

    """
    This assumes the SageMaker output path of your hpo estimator is of the format:
    output_path = f's3://{bucket}/syne-tune/{hpo_experiment_name}'

    :param tuner_name: name of a tuning experiment previously run
    :param experiment_name: name of a single experiment to load
    :param overwrite: allow deletion of local cache at SYNE_TUNE_DEFAULT_FOLDER/tuner_name
    :param return_df: whether to return a results dataframe
    :param refresh: whether to refresh plotly figure every refresh_rate seconds
    :param refresh_rate: refresh rate in seconds
    :param height: height of the plotly figure
    :param width: width of the plotly figure
    :param max_rows: max rows to display in the displayed dataframe
    :param max_cols: max columns to display in the displayed dataframe
    :param display_dataframe: whether to display the dataframe
    :param display_metadata: whether to display metadata
    :param dimensions_max_cardinality: columns with more than this number of unique values are excluded
    :param save_figure: whether to save the figure
    :return: Optional[DataFrame]
    """

    if not get_ipython() is not None:
        raise Exception("This function can only be used in a notebook")

    pd.options.display.max_columns = max_rows
    pd.options.display.max_rows = max_cols

    log = logging.getLogger("syne_tune")

    while True:
        ready_to_display = False

        # Clear cached data so that we can see the latest results
        clear_output(wait=True)
        syne_local_path = os.path.join(str(Path.home()), SYNE_TUNE_DEFAULT_FOLDER)
        job_path = os.path.join(syne_local_path, tuner_name)
        if os.path.exists(job_path) and overwrite:
            log.info("clearing local syne tune cache...")
            shutil.rmtree(job_path)

        # Download data
        tuning_experiment = load_experiment(
            tuner_name, experiment_name=experiment_name, load_tuner=True
        )
        if tuning_experiment.metadata is None:
            clear_output(wait=True)
            tuning_experiment = load_experiment(
                tuner_name, experiment_name=experiment_name, load_tuner=True
            )

        # Get metadata
        metadata = tuning_experiment.metadata
        try:
            # Get first metric
            metric = metadata["metric_names"][0]

            # Get best config
            best_config = tuning_experiment.best_config()

            ready_to_display = True
            clear_output(wait=True)

        except Exception as e:
            clear_output(wait=True)
            log.info("Waiting for tuning information to be logged...")

        if ready_to_display:
            print(f"tuner_name: {tuner_name}")
            print(f"experiment_name: {experiment_name}")
            if display_metadata:
                print(f"Metadata:\n{json.dumps(metadata, indent=4)}")

            # Filter dataframe
            results = tuning_experiment.results.sort_values(
                by=metric, ascending=False
            ).drop_duplicates("trial_id")
            trials = tuning_experiment.results.trial_id.unique()
            keep_cols = []
            for c in results.columns:
                if (
                    (metric in c)
                    or ("config_" in c)
                    or
                    # ('epoch' in c) or # how to get scheduler.resource_attr?
                    ("st_status" in c)
                    or ("trial_id" in c)
                    or ("st_decision" in c)
                ):
                    keep_cols.append(c)
            results = results[keep_cols]
            for col in results.columns:
                if len(results[col].unique()) == 1:
                    results.drop(col, inplace=True, axis=1)
            results.columns = [
                c if "config_" not in c else c.replace("config_", "")
                for c in results.columns
            ]
            results = results[[metric] + [c for c in results.columns if c != metric]]
            results.reset_index(drop=True, inplace=True)
            results = np.round(results, decimals=6)
            if display_dataframe:
                display(results)

            # Plot basic performance
            logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
            tuning_experiment.plot()

            height = max(35 * results.shape[0] if not height else height, 260)
            width = max(len(" ".join(results.columns)) * 12, 1000)
            try:  # Parallel Categories Diagram
                fig = px.parallel_categories(
                    results,
                    color=metric,
                    title=f"Syne Tune Hyperparameters ranked by {metric}",
                    color_continuous_scale=px.colors.diverging.Portland,
                    height=height,
                    width=width,
                    dimensions_max_cardinality=dimensions_max_cardinality,
                )
                fig.show()

                if save_figure:
                    fig.write_html(f"{experiment_name}_parallel_categories.html")
            except:
                log.info(f"Waiting for metric {metric} information to be logged...")

            if return_df:
                return tuning_experiment.results

        if not refresh:
            return

        time.sleep(refresh_rate)
