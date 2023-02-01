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
import pytest
import numpy as np

from syne_tune.config_space import (
    config_space_from_json_dict,
    config_space_to_json_dict,
    randint,
    lograndint,
    uniform,
    loguniform,
    choice,
    config_space_size,
    to_dict,
    from_dict,
    finrange,
    logfinrange,
    restrict_domain,
    ordinal,
    logordinal,
    OrdinalNearestNeighbor,
)


def test_convert_config_space():
    from ray.tune.search.sample import Float, Integer, Categorical
    from syne_tune.optimizer.schedulers.ray_scheduler import RayTuneScheduler

    config_space = {
        "int": randint(1, 2),
        "logint": lograndint(3, 4),
        "float": uniform(5.5, 6.5),
        "logfloat": loguniform(7.5, 8.5),
        "categorical": choice(["a", "b", "c"]),
        "const_str": "constant",
    }

    ray_config_space = RayTuneScheduler.convert_config_space(config_space)

    assert set(config_space.keys()) == set(ray_config_space.keys())
    v = ray_config_space["int"]
    # NOTE: In Ray Tune randint(lower, upper), upper is exclusive!
    assert (
        isinstance(v, Integer)
        and isinstance(v.get_sampler(), Integer._Uniform)
        and v.lower == 1
        and v.upper == 3
    )
    v = ray_config_space["logint"]
    assert (
        isinstance(v, Integer)
        and isinstance(v.get_sampler(), Integer._LogUniform)
        and v.lower == 3
        and v.upper == 5
    )
    v = ray_config_space["float"]
    assert (
        isinstance(v, Float)
        and isinstance(v.get_sampler(), Float._Uniform)
        and v.lower == 5.5
        and v.upper == 6.5
    )
    v = ray_config_space["logfloat"]
    assert (
        isinstance(v, Float)
        and isinstance(v.get_sampler(), Float._LogUniform)
        and v.lower == 7.5
        and v.upper == 8.5
    )
    v = ray_config_space["categorical"]
    assert isinstance(v, Categorical) and set(v.categories) == set(
        config_space["categorical"].categories
    )
    assert ray_config_space["const_str"] == config_space["const_str"]

    for v in config_space.values():
        if hasattr(v, "sample"):
            v.sample()


def test_serialization():
    config_space = [
        randint(1, 2),
        lograndint(3, 4),
        uniform(5.5, 6.5),
        loguniform(7.5, 8.5),
        choice(["a", "b", "c"]),
        finrange(0.0, 1.0, 4),
        finrange(0, 6, 4, cast_int=True),
        logfinrange(0.001, 1.0, 4),
        logfinrange(2, 64, 7, cast_int=True),
        ordinal([0.01, 0.05, 0.1, 0.5], kind="equal"),
        ordinal([0.01, 0.05, 0.1, 0.5], kind="nn"),
        logordinal([0.01, 0.05, 0.1, 0.5]),
    ]

    for x1 in config_space:
        x2 = from_dict(to_dict(x1))
        assert type(x1) == type(x2)
        if x1.sampler is not None:
            assert x1.sampler.__dict__ == x2.sampler.__dict__
            assert type(x1.sampler) == type(x2.sampler)
        dct1 = {
            k: v
            for k, v in x1.__dict__.items()
            if k != "sampler" and not k.startswith("_")
        }
        dct2 = {
            k: v
            for k, v in x2.__dict__.items()
            if k != "sampler" and not k.startswith("_")
        }
        assert dct1 == dct2


def test_from_and_to_json_dict():
    config_space = {
        "int": randint(1, 2),
        "logint": lograndint(3, 4),
        "float": uniform(5.5, 6.5),
        "logfloat": loguniform(7.5, 8.5),
        "categorical": choice(["a", "b", "c"]),
        "const_str": "constant",
        "const_int": 1,
        "const_float": 2.0,
    }
    assert (
        config_space_from_json_dict(config_space_to_json_dict(config_space))
        == config_space
    )


def test_config_space_size():
    upper_limit = 2**20
    config_space = {
        "a": randint(1, 6),
        "b": lograndint(1, 6),
        "c": choice(["a", "b", "c"]),
        "d": "constant",
        "e": 3.1415927,
        "g": ordinal([0.01, 0.05, 0.1, 0.5]),
    }
    cs_size = 6 * 6 * 3 * 4
    cases = [
        (config_space, cs_size),
        (dict(config_space, f=uniform(0, 1)), None),
        (dict(config_space, f=loguniform(1, 1)), cs_size),
        (dict(config_space, f=randint(3, 3)), cs_size),
        (dict(config_space, f=choice(["d"])), cs_size),
        (dict(config_space, f=randint(0, upper_limit)), None),
        (dict(config_space, f=lograndint(1, upper_limit / 10)), None),
    ]
    for cs, size in cases:
        _size = config_space_size(cs)
        assert _size == size, f"config_space_size(cs) = {_size} != {size}\n{cs}"


@pytest.mark.parametrize(
    "domain,value_set",
    [
        (finrange(0.1, 1.0, 10), np.arange(0.1, 1.1, 0.1)),
        (finrange(0.5, 1.5, 2), np.array([0.5, 1.5])),
        (logfinrange(np.exp(0.1), np.exp(1.0), 10), np.exp(np.arange(0.1, 1.1, 0.1))),
        (logfinrange(0.0001, 1.0, 5), np.array([0.0001, 0.001, 0.01, 0.1, 1.0])),
        (finrange(0, 8, 5, cast_int=True), np.array([0, 2, 4, 6, 8])),
        (
            logfinrange(8, 512, 7, cast_int=True),
            np.array([8, 16, 32, 64, 128, 256, 512]),
        ),
        (finrange(0.1, 1.0, 1), np.array([0.1])),
    ],
)
def test_finrange_domain(domain, value_set):
    seed = 31415927
    random_state = np.random.RandomState(seed)
    num_samples = 500
    sampled_values = np.array(
        domain.sample(size=num_samples, random_state=random_state)
    ).reshape((-1, 1))
    min_distances = np.min(np.abs(sampled_values - value_set.reshape((1, -1))), axis=1)
    assert np.max(min_distances) < 1e-8


@pytest.mark.parametrize(
    "domain,tp",
    [
        (uniform(0.0, 1.0), float),
        (loguniform(1.0, 10.0), float),
        (randint(0, 10), int),
        (lograndint(1, 10), int),
        (finrange(0.1, 1.0, 10), float),
        (finrange(0.1, 1.0, 1), float),
        (logfinrange(np.exp(0.1), np.exp(1.0), 10), float),
        (finrange(0, 8, 5, cast_int=True), int),
        (logfinrange(8, 512, 7, cast_int=True), int),
        (ordinal([0.01, 0.05, 0.1, 0.5]), float),
        (logordinal([1, 4, 8, 17]), int),
    ],
)
def test_type_of_sample(domain, tp):
    num_samples = 5
    seed = 31415927
    random_state = np.random.RandomState(seed)
    value = domain.sample(random_state=random_state)
    assert isinstance(value, tp), domain
    values = domain.sample(random_state=random_state, size=num_samples)
    assert isinstance(values, list) and len(values) == num_samples, domain
    assert all(isinstance(x, tp) for x in values), domain


def test_restrict_domain():
    domain = logfinrange(16, 512, 6, cast_int=True)
    assert domain._values == [16, 32, 64, 128, 256, 512]

    new_domain = restrict_domain(domain, 32, 512)
    print(new_domain)
    assert new_domain._values == [32, 64, 128, 256, 512]

    new_domain = restrict_domain(domain, 32, 128)
    print(new_domain)
    assert new_domain._values == [32, 64, 128]


@pytest.mark.parametrize(
    "categories, is_nn",
    [
        ([0, 2, 3, 5, 9], True),
        (["a", "b", "c"], False),
        ([0.1, 0.2, 0.4, 0.8, 1.6], True),
        ([0.1, 0.2, 0.4, 0.3], False),
        ([3, 2, 4, 8], False),
    ],
)
def test_ordinal_default(categories, is_nn):
    assert isinstance(ordinal(categories), OrdinalNearestNeighbor) == is_nn
