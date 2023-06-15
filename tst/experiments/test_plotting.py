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
import numpy as np
import pandas as pd
from numpy.random import RandomState

from syne_tune.constants import ST_TUNER_TIME
from syne_tune.experiments.visualization.plotting import filter_final_row_per_trial


def test_filter_final_row_per_trial():
    random_seed = 31653563
    random_state = RandomState(random_seed)
    num_seeds = 20
    nums_setups = [5, 4, 2, 5]
    grouped_dfs_all = dict()
    grouped_dfs_filtered = dict()
    for subplot_no, num_setups in enumerate(nums_setups):
        for setup_name in [f"s{s}" for s in range(num_setups)]:
            key = (subplot_no, setup_name)
            all_lst = []
            filtered_lst = []
            for seed in range(num_seeds):
                tuner_name = str(seed)
                num_trials = random_state.randint(low=5, high=50)
                trial_ids = list(range(num_trials))
                time_stamps = np.cumsum(
                    random_state.uniform(low=0.5, high=10, size=num_trials)
                )
                df_filtered = pd.DataFrame(
                    {
                        ST_TUNER_TIME: pd.Series(time_stamps),
                        "trial_id": pd.Series(trial_ids),
                    }
                )
                filtered_lst.append((tuner_name, df_filtered))
                matrices = []
                for trial_id, time_stamp in enumerate(time_stamps):
                    num_others = random_state.randint(low=0, high=6)
                    mat = np.zeros((num_others + 1, 2))
                    mat[num_others, 0] = time_stamp
                    mat[:num_others, 0] = random_state.uniform(
                        low=0.1, high=time_stamp - 0.05, size=num_others
                    )
                    mat[:, 1] = trial_id
                    matrices.append(mat)
                df_mat = np.vstack(matrices)
                sort_ind = np.argsort(df_mat[:, 0])
                df_mat = df_mat[sort_ind, :]
                df_all = pd.DataFrame(df_mat, columns=[ST_TUNER_TIME, "trial_id"])
                all_lst.append((tuner_name, df_all))
            grouped_dfs_all[key] = all_lst
            grouped_dfs_filtered[key] = filtered_lst
    grouped_dfs_filtered2 = filter_final_row_per_trial(grouped_dfs_all)
    for key, lst1 in grouped_dfs_filtered.items():
        lst2 = grouped_dfs_filtered2[key]
        assert len(lst1) == len(lst2)
        for ((seed1, df1), (seed2, df2)) in zip(lst1, lst2):
            assert seed1 == seed2
            np.testing.assert_almost_equal(df1.values, df2.values)
