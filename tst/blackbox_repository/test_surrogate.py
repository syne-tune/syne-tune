import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from syne_tune.blackbox_repository import BlackboxOffline
from syne_tune.blackbox_repository.blackbox_surrogate import add_surrogate
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular

import syne_tune.config_space as sp


np.random.seed(0)


def test_surrogate_continuous():
    surrogate = KNeighborsRegressor(n_neighbors=1)
    n = 10
    x1 = np.arange(n)
    x2 = np.arange(n)[::-1]
    cs = {
        "hp_x1": sp.randint(0, n),
        "hp_x2": sp.randint(0, n),
    }
    y = x1 * x2
    data = np.stack([x1, x2, y]).T
    df = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2", "metric_rmse"])
    blackbox = BlackboxOffline(
        df_evaluations=df, configuration_space=cs
    )
    blackbox = add_surrogate(blackbox, surrogate)

    for u, v in zip(x1, x2):
        res = blackbox.objective_function({"hp_x1": u, "hp_x2": v})
        assert res['metric_rmse'] == u * v


def test_surrogate_categorical():
    surrogate = KNeighborsRegressor(n_neighbors=1)
    n = 10
    x1 = np.arange(n)
    x2 = np.arange(n)[::-1]
    x3 = [str(i - n / 2) for i in range(n)]
    y = x1 * x2 + np.array(x3).astype(float)
    data = np.stack([x1, x2, x3, y]).T
    df = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2", "hp_x3", "metric_rmse"], dtype=float)
    df['hp_x3'] = df['hp_x3'].astype(str)
    cs = {
        "hp_x1": sp.randint(0, n),
        "hp_x2": sp.randint(0, n),
        "hp_x3": sp.choice(x3),
    }
    blackbox = BlackboxOffline(
        df_evaluations=df, configuration_space=cs
    )
    blackbox = add_surrogate(blackbox, surrogate)
    blackbox.objective_function({"hp_x1": 2, "hp_x2": 3, "hp_x3": "-2"})
    for u, v, w in zip(x1, x2, x3):
        print(u, v, w)
        res = blackbox.objective_function({"hp_x1": u, "hp_x2": v, "hp_x3": w})
        assert res['metric_rmse'] == u * v + float(w)


@pytest.mark.parametrize("surrogate", [MLPRegressor(), LinearRegression(), KNeighborsRegressor()])
def test_different_surrogates(surrogate):
    n = 10
    x1 = np.arange(n)
    x2 = np.arange(n)[::-1]
    cs = {
        "hp_x1": sp.randint(0, n),
        "hp_x2": sp.randint(0, n),
    }
    y = x1 * x2
    data = np.stack([x1, x2, y]).T
    df = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2", "metric_rmse"])
    blackbox = BlackboxOffline(
        df_evaluations=df, configuration_space=cs
    )
    blackbox = add_surrogate(blackbox, surrogate)

    for u, v in zip(x1, x2):
        blackbox.objective_function({"hp_x1": u, "hp_x2": v})


def test_blackbox_tabular_surrogate():
    n = 10
    x1 = np.arange(n)
    x2 = np.arange(n)[::-1]
    cs = {
        "hp_x1": sp.randint(0, n),
        "hp_x2": sp.randint(0, n),
    }
    n_epochs = 5
    cs_fidelity = {
        'hp_epoch': sp.randint(0, n_epochs),
    }
    data = np.stack([x1, x2]).T
    hyperparameters = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2"])
    num_seeds = 1
    num_fidelities = 2
    num_objectives = 1

    objectives_evaluations = np.random.rand(
        len(hyperparameters),
        num_seeds,
        num_fidelities,
        num_objectives
    )
    # # matches the seed, easier to test
    # for s in range(1, num_seeds):
    #     objectives_evaluations[:, s, :, :] = objectives_evaluations[:, 0, :, :]

    blackbox = BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=cs,
        fidelity_space=cs_fidelity,
        objectives_evaluations=objectives_evaluations,
    )
    surrogate = KNeighborsRegressor(n_neighbors=1)
    blackbox = add_surrogate(blackbox, surrogate=surrogate)

    for i, (u, v) in enumerate(zip(x1, x2)):
        for fidelity in range(num_fidelities):
            res = blackbox.objective_function(
                configuration={'hp_x1': u, 'hp_x2': v},
                fidelity={'hp_epoch': fidelity + 1},
            )
            print(list(res.values()), objectives_evaluations[i, 0, fidelity, :])
            assert np.allclose(
                list(res.values()),
                objectives_evaluations[i, 0, fidelity, :]
            )
