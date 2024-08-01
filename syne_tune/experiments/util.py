import logging
from typing import Dict, Tuple, List

import pandas as pd

from syne_tune.constants import ST_TUNER_TIME

DataFrameGroups = Dict[Tuple[int, str], List[Tuple[str, pd.DataFrame]]]

logger = logging.getLogger(__name__)


def filter_final_row_per_trial(grouped_dfs: DataFrameGroups) -> DataFrameGroups:
    """
    We filter rows such that only one row per trial ID remains, namely the
    one with the largest time stamp. This makes sense for single-fidelity
    methods, where reports have still been done after every epoch.
    """
    logger.info("Filtering results down to one row per trial (final result)")
    result = dict()
    for key, tuner_dfs in grouped_dfs.items():
        new_tuner_dfs = []
        for tuner_name, tuner_df in tuner_dfs:
            df_by_trial = tuner_df.groupby("trial_id")
            max_time_in_trial = df_by_trial[ST_TUNER_TIME].transform(max)
            max_time_in_trial_mask = max_time_in_trial == tuner_df[ST_TUNER_TIME]
            new_tuner_dfs.append((tuner_name, tuner_df[max_time_in_trial_mask]))
        result[key] = new_tuner_dfs
    return result
