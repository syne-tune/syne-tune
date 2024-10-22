import numpy as np
import pickle
from pathlib import Path
import os


def rand_walk_step(prev_utility):
    low = -2
    hig = 3
    new_utility = prev_utility + np.random.randint(low, hig, 3)
    # bound the utility values
    return np.minimum(np.maximum(new_utility, 1), 11)


np.random.seed(13)

starting_point = {"c_utility": np.array([7, 7, 7]), "init_level": [8.0, 6.0, 20.0]}

utilities = []
init_levels = []

cur_utilities = starting_point["c_utility"]
for _ in range(9):
    utilities.append(cur_utilities)
    # Don't change the initial level
    init_levels.append(starting_point["init_level"])

    # Update utilities for next loop iteration
    cur_utilities = rand_walk_step(cur_utilities)

context = {"c_utility": np.array(utilities), "init_level": np.array(init_levels)}

identifier = "default"
print("Generated simopt context with identifier: %s" % identifier)

cur_folder = str(Path(__file__).parent)
output_folder = cur_folder + "/generated_files"
os.makedirs(output_folder, exist_ok=True)

pickle.dump(
    context,
    open(
        output_folder + "/opt-price-random-walk-utility-context-%s.p" % identifier, "wb"
    ),
)
