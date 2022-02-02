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

from syne_tune.search_space import randint, lograndint, uniform, \
    loguniform, choice, search_space_size, randn, to_dict, from_dict, \
    finrange, logfinrange


def test_convert_config_space():
    from ray.tune.sample import Float, Integer, Categorical
    from syne_tune.optimizer.schedulers.ray_scheduler import \
        RayTuneScheduler

    config_space = {
        'int': randint(1, 2),
        'logint': lograndint(3, 4),
        'float': uniform(5.5, 6.5),
        'logfloat': loguniform(7.5, 8.5),
        'categorical': choice(['a', 'b', 'c']),
        'normal': randn(2.0, 1.0),
        'const_str': 'constant'}

    ray_config_space = RayTuneScheduler.convert_config_space(config_space)

    assert set(config_space.keys()) == set(ray_config_space.keys())
    v = ray_config_space['int']
    # NOTE: In Ray Tune randint(lower, upper), upper is exclusive!
    assert isinstance(v, Integer) and \
        isinstance(v.get_sampler(), Integer._Uniform) and \
        v.lower == 1 and v.upper == 3
    v = ray_config_space['logint']
    assert isinstance(v, Integer) and \
        isinstance(v.get_sampler(), Integer._LogUniform) and \
        v.lower == 3 and v.upper == 5
    v = ray_config_space['float']
    assert isinstance(v, Float) and \
        isinstance(v.get_sampler(), Float._Uniform) and \
        v.lower == 5.5 and v.upper == 6.5
    v = ray_config_space['logfloat']
    assert isinstance(v, Float) and \
        isinstance(v.get_sampler(), Float._LogUniform) and \
        v.lower == 7.5 and v.upper == 8.5
    v = ray_config_space['categorical']
    assert isinstance(v, Categorical) and \
        set(v.categories) == set(config_space['categorical'].categories)
    v = ray_config_space['normal']
    assert isinstance(v, Float) and isinstance(v.get_sampler(), Float._Normal) and\
           v.sampler.mean == 2.0 and v.sampler.sd == 1.0
    assert ray_config_space['const_str'] == config_space['const_str']

    for v in config_space.values():
        if hasattr(v, "sample"):
            v.sample()


def test_serialization():
    config_space = [
        randint(1, 2),
        lograndint(3, 4),
        uniform(5.5, 6.5),
        loguniform(7.5, 8.5),
        choice(['a', 'b', 'c']),
        randn(2.0, 1.0),
    ]

    for x in config_space:
        x2 = from_dict(to_dict(x))
        assert type(x) == type(x2)
        assert x.sampler.__dict__ == x2.sampler.__dict__
        assert type(x.sampler) == type(x2.sampler)
        assert {k: v for k, v in x.__dict__.items() if k != "sampler"} == {k: v for k, v in x2.__dict__.items() if k != "sampler"}


def test_search_space_size():
    upper_limit = 2 ** 20
    config_space = {
        'a': randint(1, 6),
        'b': lograndint(1, 6),
        'c': choice(['a', 'b', 'c']),
        'd': 'constant',
        'e': 3.1415927,
    }
    cs_size = 6 * 6 * 3
    cases = [
        (config_space, cs_size),
        (dict(config_space, f=uniform(0, 1)), None),
        (dict(config_space, f=loguniform(1, 1)), cs_size),
        (dict(config_space, f=randint(3, 3)), cs_size),
        (dict(config_space, f=choice(['d'])), cs_size),
        (dict(config_space, f=randint(0, upper_limit)), None),
        (dict(config_space, f=lograndint(1, upper_limit / 10)), None),
    ]
    for cs, size in cases:
        _size = search_space_size(cs)
        assert _size == size, \
            f"search_space_size(cs) = {_size} != {size}\n{cs}"


@pytest.mark.parametrize('domain,value_set', [
    (finrange(0.1, 1.0, 10),
     np.arange(0.1, 1.1, 0.1)),
    (finrange(0.5, 1.5, 2),
     np.array([0.5, 1.5])),
    (logfinrange(np.exp(0.1), np.exp(1.0), 10),
     np.exp(np.arange(0.1, 1.1, 0.1))),
    (logfinrange(0.0001, 1.0, 5),
     np.array([0.0001, 0.001, 0.01, 0.1, 1.0]))
])
def test_finrange_domain(domain, value_set):
    seed = 31415927
    random_state = np.random.RandomState(seed)
    num_samples = 500
    sampled_values = np.array(
        domain.sample(size=num_samples,
                      random_state=random_state)).reshape((-1, 1))
    min_distances = np.min(np.abs(
        sampled_values - value_set.reshape((1, -1))), axis=1)
    assert np.max(min_distances) < 1e-8


@pytest.mark.parametrize('domain,tp', [
    (uniform(0.0, 1.0), float),
    (loguniform(1.0, 10.0), float),
    (randint(0, 10), int),
    (lograndint(1, 10), int),
    (finrange(0.1, 1.0, 10), float),
    (logfinrange(np.exp(0.1), np.exp(1.0), 10), float)
])
def test_type_of_sample(domain, tp):
    num_samples = 5
    seed = 31415927
    random_state = np.random.RandomState(seed)
    value = domain.sample(random_state=random_state)
    assert isinstance(value, tp), domain
    values = domain.sample(random_state=random_state, size=num_samples)
    assert isinstance(values, list) and len(values) == num_samples, domain
    assert all(isinstance(x, tp) for x in values), domain
