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
import sys
import time
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.models.dynamnews import DynamNewsMaxProfit

plt.rcParams["font.size"] = 11

sys.path.append(str(Path(__file__).parent.parent) + "/simopt/")
sys.path.append(str(Path(__file__).parent.parent))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42

from simopt_helpers import plot_problem, evaluate_problem_price
from plotting_helper import task_pos_in_order
from experiment_master_file import experiments_meta_dict, get_task_values


print("Plotting simopt hyperparameter landscape plot from paper.")

file_dir = str(Path(__file__).parent.parent) + "/simopt/generated_files"

experiments_meta_data = experiments_meta_dict["SimOpt"]
task_values = get_task_values(experiments_meta_data)
num_probs = len(task_values)

landscape_file = "plotting/simopt_landscapes.p"

try:
    res_dict = pickle.load(open(landscape_file, "rb"))
    N = res_dict["N"]
    vmin, vmax = res_dict["vmin"], res_dict["vmax"]
    results = res_dict["results"]
except:
    print("Did not find landscape files, regenerating them now.")
    print("This will take a couple of minutes, please hang tight.")
    N = 21
    ### Initialise problems
    problems = {}
    for time_idx in range(num_probs):
        factors_file = str(file_dir) + "/default_fixed_factors.p"
        context_file = (
            str(file_dir) + "/opt-price-random-walk-utility-context-default.p"
        )

        model_fixed_factors = pickle.load(open(factors_file, "rb"))
        context_matrices = pickle.load(open(context_file, "rb"))

        # Update factors from default using the context for the given time stamp
        model_fixed_factors["c_utility"] = context_matrices["c_utility"][time_idx, :]
        model_fixed_factors["init_level"] = context_matrices["init_level"][time_idx, :]

        dyn_news_prob = DynamNewsMaxProfit(model_fixed_factors=model_fixed_factors)

        problems[time_idx] = dyn_news_prob

    rand_gen = MRG32k3a(s_ss_sss_index=[1, 2, 3])

    # Collect results for the problems
    start_t = time.time()
    results = {key: [] for key in problems}

    for ii in range(0, N):
        for jj in range(0, N):
            xx = np.array([ii, jj, 8])
            for key in problems:
                results[key].append(
                    evaluate_problem_price(problems[key], xx, rand_gen, reps=100)
                )

    vmin = np.inf
    vmax = -np.inf
    for key in problems:
        vmin = min(vmin, np.min(results[key]))
        vmax = max(vmax, np.max(results[key]))

    print("Duration: %s" % np.round(time.time() - start_t, 3))
    pickle.dump(
        {"results": results, "vmin": vmin, "vmax": vmax, "N": N},
        open(landscape_file, "wb"),
    )

# Plot the problems
fig, ax = plt.subplots(1, 5, figsize=(8, 2 * 8 / 11))

ii = 0
for key in [0, 1, 4, 6, 8]:
    task_num = task_pos_in_order(key, task_values)
    cb = plot_problem(
        ax=ax[ii],
        ii=ii,
        results=results[key],
        task_num=task_num,
        N=N,
        num_cols=5,
        vmin=vmin,
        vmax=vmax,
    )
    if ii != 0:
        ax[ii].set_yticklabels([])
    ii += 1
fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
cbar = fig.colorbar(cb, cax=cbar_ax)
cbar.set_label("Profit")

plot_file = "plotting/Figures/simopt_landscape.pdf"
plt.savefig(plot_file, bbox_inches="tight", dpi=400, pad_inches=0)
print("Generated file %s" % plot_file)
