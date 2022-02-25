import tempfile

import numpy as np
import pandas as pd

import syne_tune.config_space as sp

from benchmarking.blackbox_repository import BlackboxOffline
from benchmarking.blackbox_repository.blackbox import from_function
from benchmarking.blackbox_repository.blackbox_tabular import BlackboxTabular
from benchmarking.blackbox_repository.blackbox_offline import deserialize as deserialize_offline
from benchmarking.blackbox_repository.blackbox_tabular import deserialize as deserialize_tabular
from benchmarking.blackbox_repository.blackbox_offline import serialize as serialize_offline
from benchmarking.blackbox_repository.blackbox_tabular import serialize as serialize_tabular


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

def test_blackbox_from_function():
    def eval_fun(config, fidelity, seed):
        return {'metric_rmse': config['hp_x1'] * config['hp_x2']}
    blackbox = from_function(configuration_space=cs, eval_fun=eval_fun)
    for u, v in zip(x1, x2):
        res = blackbox.objective_function({"hp_x1": u, "hp_x2": v})
        assert res['metric_rmse'] == u * v


def test_blackbox_dataframe_call():

    y = x1 * x2
    data = np.stack([x1, x2, y]).T
    df = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2", "metric_rmse"])
    blackbox = BlackboxOffline(df_evaluations=df, configuration_space=cs)

    for u, v in zip(x1, x2):
        res = blackbox.objective_function({"hp_x1": u, "hp_x2": v})
        assert res['metric_rmse'] == u * v


def test_blackbox_fidelity():

    # build dummy values for fidelities
    fidelities = []
    for fidelity in range(n_epochs):
        dummy_y = x1 * x2 + fidelity
        fidelity_vec = np.ones_like(x1) * fidelity
        fidelities.append(np.stack([x1, x2, fidelity_vec, dummy_y]).T)
    data = np.vstack(fidelities)

    df = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2", "hp_epoch", "metric_rmse"])

    blackbox = BlackboxOffline(df_evaluations=df, configuration_space=cs, fidelity_space=cs_fidelity)

    for u, v in zip(x1, x2):
        for epoch in range(n_epochs):
            res = blackbox.objective_function({"hp_x1": u, "hp_x2": v}, {'hp_epoch': epoch})
            assert res['metric_rmse'] == u * v + epoch

    # check that blackbox can be called with configuration instead of dict
    config = {k: v.sample() for k, v in blackbox.configuration_space.items()}
    config['hp_x1'] = u
    config['hp_x2'] = v
    res = blackbox.objective_function(config, {'hp_epoch': epoch})
    assert res['metric_rmse'] == u * v + epoch

    # check that blackbox can be called with fidelity value instead of dict
    config = {k: v.sample() for k, v in blackbox.configuration_space.items()}
    config['hp_x1'] = u
    config['hp_x2'] = v
    res = blackbox.objective_function(config, epoch)
    assert res['metric_rmse'] == u * v + epoch

    # check that blackbox can be called with fidelity value instead of dict
    config = {k: v.sample() for k, v in blackbox.configuration_space.items()}
    config['hp_x1'] = u
    config['hp_x2'] = v
    res = blackbox(config, epoch)
    assert res['metric_rmse'] == u * v + epoch


def test_blackbox_seed():
    # build dummy values for seeds
    n_seeds = 4
    seeds = []
    for seed in range(n_seeds):
        dummy_y = x1 * x2 + seed
        seed_vec = np.ones_like(x1) * seed
        seeds.append(np.stack([x1, x2, seed_vec, dummy_y]).T)
    data = np.vstack(seeds)

    df = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2", "seed", "metric_rmse"])
    blackbox = BlackboxOffline(
        df_evaluations=df, configuration_space=cs, seed_col="seed"
    )

    for u, v in zip(x1, x2):
        for seed in range(n_seeds):
            res = blackbox.objective_function({"hp_x1": u, "hp_x2": v}, seed=seed)
            assert res['metric_rmse'] == u * v + seed


def test_blackbox_offline_serialization():
    y = x1 * x2
    data = np.stack([x1, x2, y]).T
    df = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2", "metric_rmse"])

    blackbox = BlackboxOffline(df_evaluations=df, configuration_space=cs)

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"serializing and deserializing blackbox in folder {tmpdirname}")
        serialize_offline({"task": blackbox}, tmpdirname)
        blackbox_deserialized = deserialize_offline(tmpdirname)['task']
        for u, v in zip(x1, x2):
            res = blackbox_deserialized.objective_function({"hp_x1": u, "hp_x2": v})
            assert res['metric_rmse'] == u * v


def test_blackbox_offline_fidelities():
    data = np.concatenate(
        [np.stack([x1, x2,       x1 * x2,   np.ones_like(x1, dtype=np.int)], axis=1),
         np.stack([x1, x2, 0.5 * x1 * x2, 2*np.ones_like(x1, dtype=np.int)], axis=1)],
        axis=0)
    df = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2", "metric_rmse", "step"])

    blackbox = BlackboxOffline(df_evaluations=df, configuration_space=cs,
                               fidelity_space=dict(step=sp.randint(1, 2)))

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"serializing and deserializing blackbox in folder {tmpdirname}")
        for u, v in zip(x1, x2):
            res = blackbox.objective_function({"hp_x1": u, "hp_x2": v}, fidelity=1)
            assert res['metric_rmse'] == u * v
            
            res = blackbox.objective_function({"hp_x1": u, "hp_x2": v}, fidelity=2)
            assert res['metric_rmse'] == 0.5 * u * v
            
            res = blackbox.objective_function({"hp_x1": u, "hp_x2": v}, fidelity=None)
            # Returns a tensor with shape (num_fidelities, num_objectives)
            assert res.shape == (2, 1)
            assert (res == np.array([u * v, 0.5 * u * v]).reshape(2, 1)).all()


def test_blackbox_tabular_serialization():
    hyperparameters = pd.DataFrame(data=np.stack([x1, x2]).T, columns=["hp_x1", "hp_x2"])
    num_seeds = 1
    num_fidelities = 2
    num_objectives = 1

    def make_dummy_blackbox():
        objectives_evaluations = np.random.rand(
            len(hyperparameters),
            num_seeds,
            num_fidelities,
            num_objectives
        )
        return BlackboxTabular(
            hyperparameters=hyperparameters,
            configuration_space=cs,
            fidelity_space=cs_fidelity,
            objectives_evaluations=objectives_evaluations,
        )

    bb_dict = {
        "protein": make_dummy_blackbox(),
        "slice": make_dummy_blackbox(),
    }

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"serializing and deserializing blackbox in folder {tmpdirname}")
        serialize_tabular(bb_dict, tmpdirname)
        bb_dict2 = deserialize_tabular(tmpdirname)

        print(bb_dict2['slice'].objective_function({"hp_x1": x1[0], "hp_x2": x2[0]}, fidelity={'hp_epochs': 1}))

        for key in bb_dict2.keys():
            bb1 = bb_dict[key]
            bb2 = bb_dict2[key]
            # assert sp.equal(bb1.configuration_space, bb2.configuration_space)
            # assert sp.equal(bb1.fidelity_space, bb2.fidelity_space)
            assert np.all(bb1.fidelity_values == bb2.fidelity_values)
            assert bb1.objectives_names == bb2.objectives_names
            np.testing.assert_allclose(
                bb1.objectives_evaluations.reshape(-1),
                bb2.objectives_evaluations.reshape(-1)
            )


        #blackbox.serialize(tmpdirname)
        #blackbox_deserialized = deserialize(tmpdirname)
        #for u, v in zip(x1, x2):
        #    res = blackbox_deserialized.objective_function({"hp_x1": u, "hp_x2": v})
        #    assert res['metric_rmse'] == u * v



def test_blackbox_tabular():
    data = np.stack([x1, x2]).T
    hyperparameters = pd.DataFrame(data=data, columns=["hp_x1", "hp_x2"])
    num_seeds = 3
    num_fidelities = 5
    num_objectives = 2

    objectives_evaluations = np.random.rand(
        len(hyperparameters),
        num_seeds,
        num_fidelities,
        num_objectives
    )

    blackbox = BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=cs,
        fidelity_space=cs_fidelity,
        objectives_evaluations=objectives_evaluations,
        objectives_names=["a", "b"],
    )

    for i, (u, v) in enumerate(zip(x1, x2)):
        res = blackbox.objective_function(
            configuration={'hp_x1': u, 'hp_x2': v},
            fidelity={'hp_epoch': num_fidelities},
            seed=num_seeds - 1
        )
        assert np.allclose(
            list(res.values()),
            objectives_evaluations[i, num_seeds - 1, num_fidelities - 1, :]
        )
