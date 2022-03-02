import syne_tune.config_space as cs
import numpy as np

from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.schedulers.botorch.botorch_gp import BotorchGP


def test_featurize():
    config_space = {
        "steps": 100,
        "x": cs.randint(0, 20),
        "y": cs.uniform(0, 1),
        "z": cs.choice(["a", "b", "c"]),
    }

    categorical_maps = {
        k: {cat: i for i, cat in enumerate(v.categories)}
        for k, v in config_space.items()
        if isinstance(v, cs.Categorical)
    }
    inv_categorical_maps = {hp: dict(zip(map.values(), map.keys())) for hp, map in categorical_maps.items()}
    np.random.seed(0)
    for _ in range(3):
        config = {
            k: v.sample()
            if isinstance(v, cs.Domain) else v
            for k, v in config_space.items()
        }
        feature_vector = BotorchGP._encode_config(config_space, config, categorical_maps)

        config_reconstructed = BotorchGP._decode_config(
            config_space, encoded_vector=feature_vector, inv_categorical_maps=inv_categorical_maps
        )
        assert config == config_reconstructed


def test_calls():
    config_space = {
        "steps": 100,
        "x": cs.randint(0, 20),
        "y": cs.uniform(0, 1),
        "z": cs.choice(["a", "b", "c"]),
    }
    np.random.seed(0)

    metric = "objective"
    scheduler = BotorchGP(config_space=config_space, mode="max", metric=metric, num_init_random_draws=2)

    trials = []
    for i in range(10):
        suggestion = scheduler.suggest(i)
        trials.append(Trial(trial_id=i, config=suggestion.config, creation_time=None))

    make_metric = lambda x: {metric: x}

    scheduler.on_trial_result(trials[0], make_metric(0.0))
    scheduler.on_trial_result(trials[2], make_metric(2.0))
    scheduler.on_trial_result(trials[3], make_metric(3.0))
    scheduler.on_trial_complete(trials[3], make_metric(3.0))
    i += 1
    suggestion = scheduler.suggest(i)
    trials.append(Trial(trial_id=i, config=suggestion.config, creation_time=None))

    scheduler.on_trial_complete(trials[0], make_metric(0.0))
    i += 1
    suggestion = scheduler.suggest(i)
    trials.append(Trial(trial_id=i, config=suggestion.config, creation_time=None))

    scheduler.on_trial_complete(trials[1], make_metric(1.0))
    i += 1
    suggestion = scheduler.suggest(i)
    trials.append(Trial(trial_id=i, config=suggestion.config, creation_time=None))

