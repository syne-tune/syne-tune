import pickle
from pathlib import Path
import os

cur_folder = str(Path(__file__).parent)
output_folder = cur_folder + "/generated_files"
os.makedirs(output_folder, exist_ok=True)

# Store default fixed factors
pickle.dump(
    {
        "mu": 1,
        "num_customer": 30,
        "num_prod": 3,
        "c_utility": [6, 8, 10],
        "init_level": [8.0, 6.0, 20.0],
        "price": [9.0, 9.0, 9.0],
        "cost": [5.0, 5.0, 5.0],
    },
    open(output_folder + "/default_fixed_factors.p", "wb"),
)
