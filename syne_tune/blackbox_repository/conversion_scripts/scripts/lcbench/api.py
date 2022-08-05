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
# File taken from LCBench to avoid having to install a directory manually
# https://github.com/automl/LCBench

import os as os
import numpy as np
import json
import pickle
import gzip


class Benchmark:
    """API for TabularBench."""

    def __init__(self, data_dir, cache=False, cache_dir="cached/"):
        """Initialize dataset (will take a few seconds-minutes).

        Keyword arguments:
        bench_data -- str, the raw benchmark data directory
        """
        if not os.path.isfile(data_dir) or not data_dir.endswith(".json"):
            raise ValueError("Please specify path to the bench json file.")

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache = cache

        print("==> Loading data...")
        self.data = self._read_data(data_dir)
        self.dataset_names = list(self.data.keys())
        print("==> Done.")

    def query(self, dataset_name, tag, config_id):
        """Query a run.

        Keyword arguments:
        dataset_name -- str, the name of the dataset in the benchmark
        tag -- str, the tag you want to query
        config_id -- int, an identifier for which run you want to query, if too large will query the last run
        """
        config_id = str(config_id)
        if dataset_name not in self.dataset_names:
            raise ValueError("Dataset name not found.")

        if config_id not in self.data[dataset_name].keys():
            raise ValueError(
                "Config nr %s not found for dataset %s." % (config_id, dataset_name)
            )

        if tag in self.data[dataset_name][config_id]["log"].keys():
            return self.data[dataset_name][config_id]["log"][tag]

        if tag in self.data[dataset_name][config_id]["results"].keys():
            return self.data[dataset_name][config_id]["results"][tag]

        if tag in self.data[dataset_name][config_id]["config"].keys():
            return self.data[dataset_name][config_id]["config"][tag]

        if tag == "config":
            return self.data[dataset_name][config_id]["config"]

        raise ValueError(
            "Tag %s not found for config %s for dataset %s"
            % (tag, config_id, dataset_name)
        )

    def query_best(self, dataset_name, tag, criterion, position=0):
        """Query the n-th best run. "Best" here means achieving the largest value at any epoch/step,

        Keyword arguments:
        dataset_name -- str, the name of the dataset in the benchmark
        tag -- str, the tag you want to query
        criterion -- str, the tag you want to use for the ranking
        position -- int, an identifier for which position in the ranking you want to query
        """
        performances = []
        for config_id in self.data[dataset_name].keys():
            performances.append(
                (config_id, max(self.query(dataset_name, criterion, config_id)))
            )

        performances.sort(key=lambda x: x[1] * 1000, reverse=True)
        desired_position = performances[position][0]

        return self.query(dataset_name, tag, desired_position)

    def get_queriable_tags(self, dataset_name=None, config_id=None):
        """Returns a list of all queriable tags"""
        if dataset_name is None or config_id is None:
            dataset_name = list(self.data.keys())[0]
            config_id = list(self.data[dataset_name].keys())[0]
        else:
            config_id = str(config_id)
        log_tags = list(self.data[dataset_name][config_id]["log"].keys())
        result_tags = list(self.data[dataset_name][config_id]["results"].keys())
        config_tags = list(self.data[dataset_name][config_id]["config"].keys())
        additional = ["config"]
        return log_tags + result_tags + config_tags + additional

    def get_dataset_names(self):
        """Returns a list of all availabe dataset names like defined on openml"""
        return self.dataset_names

    def get_openml_task_ids(self):
        """Returns a list of openml task ids"""
        task_ids = []
        for dataset_name in self.dataset_names:
            task_ids.append(self.query(dataset_name, "OpenML_task_id", 1))
        return task_ids

    def get_number_of_configs(self, dataset_name):
        """Returns the number of configurations for a dataset"""
        if dataset_name not in self.dataset_names:
            raise ValueError("Dataset name not found.")
        return len(self.data[dataset_name].keys())

    def get_config(self, dataset_name, config_id):
        """Returns the configuration of a run specified by dataset name and config id"""
        if dataset_name not in self.dataset_names:
            raise ValueError("Dataset name not found.")
        return self.data[dataset_name][config_id]["config"]

    def plot_by_name(
        self,
        dataset_names,
        x_col,
        y_col,
        n_configs=10,
        show_best=False,
        xscale="linear",
        yscale="linear",
        criterion=None,
    ):
        """Plot multiple datasets and multiple runs.

        Keyword arguments:
        dataset_names -- list
        x_col -- str, tag to plot on x-axis
        y_col -- str, tag to plot on y-axis
        n_configs -- int, number of configs to plot for each dataset
        show_best -- bool, weather to show the n_configs best (according to query_best())
        xscale -- str, set xscale, options as in matplotlib: "linear", "log", "symlog", "logit", ...
        yscale -- str, set yscale, options as in matplotlib: "linear", "log", "symlog", "logit", ...
        criterion -- str, tag used as criterion for query_best()
        """
        import matplotlib.pyplot as plt

        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if not isinstance(dataset_names, (list, np.ndarray)):
            raise ValueError(
                "Please specify a dataset name or a list list of dataset names."
            )

        n_rows = len(dataset_names)
        fig, axes = plt.subplots(
            n_rows, 1, sharex=False, sharey=False, figsize=(10, 7 * n_rows)
        )

        if criterion is None:
            criterion = y_col

        loop_arg = enumerate(axes.flatten()) if len(dataset_names) > 1 else [(0, axes)]

        for ind_ax, ax in loop_arg:
            for ind in range(n_configs):
                try:
                    if ind == 0:
                        instances = int(
                            self.query(dataset_names[ind_ax], "instances", 0)
                        )
                        classes = int(self.query(dataset_names[ind_ax], "classes", 0))
                        features = int(self.query(dataset_names[ind_ax], "features", 0))

                    if show_best:
                        x = self.query_best(
                            dataset_names[ind_ax], x_col, criterion, ind
                        )
                        y = self.query_best(
                            dataset_names[ind_ax], y_col, criterion, ind
                        )
                    else:
                        x = self.query(dataset_names[ind_ax], x_col, ind + 1)
                        y = self.query(dataset_names[ind_ax], y_col, ind + 1)

                    ax.plot(x, y, "p-")
                    ax.set_xscale(xscale)
                    ax.set_yscale(yscale)
                    ax.set(xlabel="step", ylabel=y_col)
                    title_str = ", ".join(
                        [
                            dataset_names[ind_ax],
                            "features: " + str(features),
                            "classes: " + str(classes),
                            "instances: " + str(instances),
                        ]
                    )
                    ax.title.set_text(title_str)
                except ValueError:
                    print(
                        "Run %i not found for dataset %s" % (ind, dataset_names[ind_ax])
                    )
                except Exception as e:
                    raise e

    def _cache_data(self, data, cache_file):
        os.makedirs(self.cache_dir, exist_ok=True)
        with gzip.open(cache_file, "wb") as f:
            pickle.dump(data, f)

    def _read_cached_data(self, cache_file):
        with gzip.open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data

    def _read_file_string(self, path):
        """Reads a large json string from path. Python file handler has issues with large files so it has to be chunked."""
        # Shoutout to https://stackoverflow.com/questions/48122798/oserror-errno-22-invalid-argument-when-reading-a-huge-file
        file_str = ""
        with open(path, "r") as f:
            while True:
                block = f.read(64 * (1 << 20))  # Read 64 MB at a time
                if not block:  # Reached EOF
                    break
                file_str += block
        return file_str

    def _read_data(self, path):
        """Reads cached data if available. If not, reads json and caches the data as .pkl.gz"""
        cache_file = os.path.join(
            self.cache_dir, os.path.basename(self.data_dir).replace(".json", ".pkl.gz")
        )
        if os.path.exists(cache_file) and self.cache:
            print("==> Found cached data, loading...")
            data = self._read_cached_data(cache_file)
        else:
            print("==> No cached data found or cache set to False.")
            print("==> Reading json data...")
            data = json.loads(self._read_file_string(path))
            if self.cache:
                print("==> Caching data...")
                self._cache_data(data, cache_file)
        return data
