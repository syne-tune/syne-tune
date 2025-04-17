import XGBoost_helper
import pickle
import numpy as np

gen_seed = 13
num_hyp_pars = 1000

np.random.seed(gen_seed)

# Random ####
learning_rates = [
    XGBoost_helper.make_sample(
        samp_type="continuous", min_lim=1e-6, max_lim=1, scale="log"
    )
    for _ in range(num_hyp_pars)
]

min_child_weight = [
    XGBoost_helper.make_sample(
        samp_type="continuous", min_lim=1e-6, max_lim=32, scale="log"
    )
    for _ in range(num_hyp_pars)
]

max_depth = [
    XGBoost_helper.make_sample(samp_type="integer", min_lim=2, max_lim=32, scale="log")
    for _ in range(num_hyp_pars)
]

n_estimators = [
    XGBoost_helper.make_sample(samp_type="integer", min_lim=2, max_lim=256, scale="log")
    for _ in range(num_hyp_pars)
]

parameters_mat_rand = {
    "learning_rates": learning_rates,
    "min_child_weight": min_child_weight,
    "max_depth": max_depth,
    "n_estimators": n_estimators,
}

pickle.dump(
    parameters_mat_rand,
    open(
        "hyperparameters_file_random_num-%s_seed-%s.p" % (num_hyp_pars, gen_seed), "wb"
    ),
)
