import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from benchmarking.blackbox_repository import load, add_surrogate
from syne_tune.util import catchtime

bb = load("fcnet")["slice_localization"]
config_space = bb.configuration_space
hp_name = 'hp_init_lr'
# hp_name = "hp_n_units_1"
config_space_hp_values = bb.configuration_space[hp_name].categories
hp_values = np.linspace(min(config_space_hp_values), max(config_space_hp_values))

hp = {k: v.sample() for k, v in config_space.items()}
hp = {
    'hp_activation_fn_1': 'tanh',
    'hp_activation_fn_2': 'relu',
    'hp_batch_size': 16,
    'hp_dropout_1': 0.6,
    'hp_dropout_2': 0.6,
    'hp_init_lr': 0.05,
    'hp_lr_schedule': 'cosine',
    'hp_n_units_1': 256,
    'hp_n_units_2': 256
}

with catchtime("fit"):
    surrogates = {
        'knn': add_surrogate(bb, surrogate=KNeighborsRegressor(n_neighbors=1), max_fit_samples=None),
        # 'linear': add_surrogate(bb, surrogate=LinearRegression()),
        # 'mlp': add_surrogate(bb, surrogate=MLPRegressor()),
    }



def eval_hp(bb, learning_rate):
    hp[hp_name] = learning_rate
    return bb(hp, fidelity=100, seed=0)['metric_valid_loss']

with catchtime("predict"):
    for name, bb_surrogate in surrogates.items():
        ys = []
        for hp_value in hp_values:
            ys.append(eval_hp(bb_surrogate, hp_value))
        plt.plot(
            hp_values,
            ys,
            label=name,
            marker='+'
        )

plt.plot(
    config_space_hp_values,
    [eval_hp(bb, lr) for lr in config_space_hp_values],
    label="true",
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("learning rate")
plt.ylabel("metric validation loss")
plt.legend()
plt.show()
