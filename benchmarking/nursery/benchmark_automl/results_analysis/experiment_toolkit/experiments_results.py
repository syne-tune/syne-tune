import json
from typing import List, Callable, Tuple, Optional, Set

import pandas as pd
from syne_tune.util import experiment_path


def load_results_by_experiment_names(
        experiment_names: Tuple[str, ...],
        metadata_to_setup: Callable[[dict], Optional[str]],
        metadata_print_values: Optional[List[str]] = None,
        metadata_to_subplot: Optional[Callable[[dict], Optional[int]]] = None,
        verbose: bool = False) -> (Optional[pd.DataFrame], Set[str]):
    """
    Loads results into a dataframe similar to `load_experiments_task`, but with
    some differences. First, tuner names are selected by their prefixes (in
    `experiment_names`).

    Second, grouping and filtering is done only w.r.t. meta, so that only
    metadata.json and results.csv.zip are needed. This works best if experiments
    have been launched with launch_hpo.py. `metadata_to_setup` maps a metadata
    dict to a setup name or None. Runs mapped to None are to be filtered out,
    while runs mapped to the same setup name are samples for that setup, giving
    rise to statistics which are plotted for that setup. Here, multiple setups
    are compared against each other in the same plot.

    `metadata_to_subplot` is optional. If given, it serves as grouping into
    subplots. Runs mapped to None are filtered out as well.

    :param experiment_names:
    :param metadata_to_setup:
    :param metadata_print_values: If given, list of metadata keys. Values for
        these keys are collected from metadata, and lists are printed
    :param metadata_to_subplot:
    :return:
    """
    dfs = []
    setup_names = set()
    if metadata_print_values is None:
        metadata_print_values = []
    for experiment_name in experiment_names:
        pattern = experiment_name + "*/metadata.json"
        print(f"pattern = {pattern}")
        metadata_values = {k: dict() for k in metadata_print_values}
        for meta_path in experiment_path().rglob(pattern):
            tuner_path = meta_path.parent
            try:
                with open(str(meta_path), "r") as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                metadata = None
            if metadata is not None:
                try:
                    setup_name = metadata_to_setup(metadata)
                    if metadata_to_subplot is not None:
                        subplot_no = metadata_to_subplot(metadata)
                    else:
                        subplot_no = 0
                except BaseException as err:
                    print(f"Caught exception for {tuner_path}:\n" + str(err))
                    raise
                if setup_name is not None and subplot_no is not None:
                    try:
                        if (tuner_path / "results.csv.zip").exists():
                            df = pd.read_csv(tuner_path / "results.csv.zip")
                        else:
                            df = pd.read_csv(tuner_path / "results.csv")
                    except FileNotFoundError:
                        df = None
                    if df is None:
                        if verbose:
                           print(f"{tuner_path}: Meta-data matches filter, but "
                              "results file not found. Skipping.")
                    else:
                        df['setup_name'] = setup_name
                        setup_names.add(setup_name)
                        if metadata_to_subplot is not None:
                            df['subplot_no'] = subplot_no
                        df['tuner_name'] = tuner_path.name
                        # Add all metadata fields to the dataframe
                        for k, v in metadata.items():
                            if isinstance(v, list):
                                for i, vi in enumerate(v):
                                    df[k + '_%d' % i] = vi
                            else:
                                df[k] = v
                        dfs.append(df)
                        for name in metadata_print_values:
                            if name in metadata:
                                dct = metadata_values[name]
                                value = metadata[name]
                                if setup_name in dct:
                                    dct[setup_name].append(value)
                                else:
                                    dct[setup_name] = [value]
        if metadata_print_values:
            parts = []
            for name in metadata_print_values:
                dct = metadata_values[name]
                parts.append(f"{name}:")
                for setup_name, values in dct.items():
                    parts.append(f"  {setup_name}: {values}")
            print('\n'.join(parts))

    if not dfs:
        res_df = None
    else:
        res_df = pd.concat(dfs, ignore_index=True)
    return res_df, setup_names