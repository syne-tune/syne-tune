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

from collections import Counter
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from pytest import approx

from syne_tune.config_space import uniform, randint, choice, loguniform, \
    lograndint, finrange, logfinrange, reverseloguniform
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_impl \
    import HyperparameterRangesImpl
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext \
    import ExtendedConfiguration


def _assert_allclose_config(c1, c2, hp_ranges):
    c1_tpl = hp_ranges.config_to_tuple(c1)
    c2_tpl = hp_ranges.config_to_tuple(c2)
    assert_allclose(c1_tpl, c2_tpl)


@pytest.mark.parametrize('lower,upper,external_hp,internal_ndarray,domain,size', [
    (0.0, 8.0, 0.0, 0.0, uniform, None),
    (0.0, 8.0, 8.0, 1.0, uniform, None),
    (0.0, 8.0, 2.0, 0.25, uniform, None),
    (100.2, 100.6, 100.4, 0.5, uniform, None),
    (-2.0, 8.0, 0.0, 0.2, uniform, None),
    (-11.0, -1.0, -10.0, 0.1, uniform, None),
    (1.0, 8.0, 1.0, 0.0, loguniform, None),
    (1.0, 8.0, 8.0, 1.0, loguniform, None),
    (1.0, 10000.0, 10.0, 0.25, loguniform, None),
    (1.0, 10000.0, 100.0, 0.5, loguniform, None),
    (1.0, 10000.0, 1000.0, 0.75, loguniform, None),
    (0.001, 0.1, 0.01, 0.5, loguniform, None),
    (0.1, 100, 1.0, 1.0/3, loguniform, None),
    (0.5, 0.99, 0.5, 0.0, reverseloguniform, None),
    (0.5, 0.99, 0.99, 1.0, reverseloguniform, None),
    (0.9, 0.99999, 0.99, 0.25, reverseloguniform, None),
    (0.9, 0.99999, 0.999, 0.5, reverseloguniform, None),
    (0.9, 0.99999, 0.9999, 0.75, reverseloguniform, None),
    (0.5, 15.0/16.0, 0.75, 1.0/3, reverseloguniform, None),
    (0.5, 15.0/16.0, 7.0/8.0, 2.0/3, reverseloguniform, None),
    (1, 10001, 5001, 0.5, randint, None),
    (-10, 10, 0, 0.5, randint, None),
    (0.1, 1.0, 0.1, 0.5, finrange, 1),
    (0.1, 1.0, 0.1, 0.5/10, finrange, 10),
    (0.1, 1.0, 1.0, 9.5/10, finrange, 10),
    (0.1, 1.0, 0.5, 4.5/10, finrange, 10),
    (np.exp(0.1), np.exp(1.0), np.exp(0.1), 0.5/10, logfinrange, 10),
    (np.exp(0.1), np.exp(1.0), np.exp(1.0), 9.5/10, logfinrange, 10),
    (np.exp(0.1), np.exp(1.0), np.exp(0.5), 4.5/10, logfinrange, 10)
])
def test_continuous_to_and_from_ndarray(
        lower, upper, external_hp, internal_ndarray, domain, size):
    if size is None:
        hp_range = domain(lower, upper)
    else:
        hp_range = domain(lower, upper, size=size)
    hp_ranges = make_hyperparameter_ranges({'a': hp_range})
    config = hp_ranges.tuple_to_config((external_hp,))
    config_enc = np.array([internal_ndarray])
    assert_allclose(hp_ranges.to_ndarray(config), config_enc)
    _assert_allclose_config(
        hp_ranges.from_ndarray(config_enc), config, hp_ranges)


@pytest.mark.parametrize('choices,external_hp,internal_ndarray', [
    (['a', 'b'], 'a', [0.25]),
    (['a', 'b'], 'b', [0.75]),
    (['a', 'b', 'c'], 'b', [0.0, 1.0, 0.0]),
    (['a', 'b', 'c', 'd'], 'c', [0.0, 0.0, 1.0, 0.0]),
])
def test_categorical_to_and_from_ndarray(choices, external_hp, internal_ndarray):
    hp_ranges = make_hyperparameter_ranges({'a': choice(choices)})
    config = hp_ranges.tuple_to_config((external_hp,))
    config_enc = np.array(internal_ndarray)
    assert_allclose(hp_ranges.to_ndarray(config), config_enc)
    assert hp_ranges.from_ndarray(config_enc) == config


# Going to internal representation and back should give back the original value
@pytest.mark.parametrize('lower,upper,domain', [
    (0.0, 8.0, uniform),
    (0.01, 0.1, uniform),
    (-10.0, -5.1, uniform),
    (-1000000000000000.0, 100000000000000000.0, uniform),
    (10.0, 10000000000.0, loguniform),
    (-1000.0, 100.0, uniform),
    (1.0, 1000.0, loguniform),
    (10.0, 15.0, loguniform),
    (0.1, 20.0, loguniform),
    (0.0, 0.99, reverseloguniform),
    (0.999999999, 0.99999999999, reverseloguniform),
])
def test_continuous_to_ndarray_and_back(lower, upper, domain):
    # checks the lower bound upper bound and 10 random values
    _test_to_ndarray_and_back(lower, upper, lower, domain)
    _test_to_ndarray_and_back(lower, upper, upper, domain)
    rnd = np.random.RandomState(0)
    for random_hp in rnd.uniform(lower, upper, size=10):
        _test_to_ndarray_and_back(lower, upper, random_hp, domain)


# helper for the previous test
def _test_to_ndarray_and_back(lower, upper, external_hp, domain):
    hp_ranges = make_hyperparameter_ranges({'a': domain(lower, upper)})
    config = hp_ranges.tuple_to_config((external_hp,))
    assert hp_ranges.from_ndarray(
        hp_ranges.to_ndarray(config))['a'] == approx(external_hp)


@pytest.mark.parametrize('lower,upper,domain', [
    (0, 8, randint),
    (1, 20, randint),
    (-10, -5, randint),
    (-1000000000000000, 100000000000000000, randint),
    (10, 10000000000, lograndint),
    (-1000, 100, randint),
    (1, 1000, lograndint),
    (10, 15, lograndint),
])
def test_integer_to_ndarray_and_back(lower, upper, domain):
    # checks the lower bound upper bound and 15 random values
    _test_to_ndarray_and_back(lower, upper, lower, domain)
    _test_to_ndarray_and_back(lower, upper, upper, domain)
    rnd = np.random.RandomState(0)
    for random_hp in rnd.randint(lower + 1, upper, size=15):
        _test_to_ndarray_and_back(lower, upper, int(random_hp), domain)


# this is more of a functional test testing of HP conversion and scaling
# it generates random candidates and checks the distribution is correct
# and also that they can be transformed to internal representation and back while still obtaining
# the same value
def test_distribution_of_random_candidates():
    random_state = np.random.RandomState(0)
    hp_ranges = make_hyperparameter_ranges({
        '0': uniform(1.0, 1000.0),
        '1': loguniform(1.0, 1000.0),
        '2': reverseloguniform(0.9, 0.9999),
        '3': randint(1, 1000),
        '4': lograndint(1, 1000),
        '5': choice(['a', 'b', 'c'])})
    num_random_candidates = 600
    random_candidates = hp_ranges.random_configs(
        random_state, num_random_candidates)

    # check converting back gets to the same candidate
    for cand in random_candidates[2:]:
        cand_tpl = hp_ranges.config_to_tuple(cand)
        ndarray_candidate = hp_ranges.to_ndarray(cand)
        converted_back = hp_ranges.from_ndarray(ndarray_candidate)
        back_tpl = hp_ranges.config_to_tuple(converted_back)
        for hp, hp_converted_back in zip(cand_tpl, back_tpl):
            if isinstance(hp, str):
                assert hp == hp_converted_back
            else:
                assert_almost_equal(hp, hp_converted_back)

    hps0, hps1, hps2, hps3, hps4, hps5 = zip(*[
        hp_ranges.config_to_tuple(x) for x in random_candidates])
    assert 200 < np.percentile(hps0, 25) < 300
    assert 450 < np.percentile(hps0, 50) < 550
    assert 700 < np.percentile(hps0, 75) < 800

    # same bounds as the previous but log scaling
    assert 3 < np.percentile(hps1, 25) < 10
    assert 20 < np.percentile(hps1, 50) < 40
    assert 100 < np.percentile(hps1, 75) < 250

    # reverse log
    assert 0.9 < np.percentile(hps2, 25) < 0.99
    assert 0.99 < np.percentile(hps2, 50) < 0.999
    assert 0.999 < np.percentile(hps2, 75) < 0.9999

    # integer
    assert 200 < np.percentile(hps3, 25) < 300
    assert 450 < np.percentile(hps3, 50) < 550
    assert 700 < np.percentile(hps3, 75) < 800

    # same bounds as the previous but log scaling
    assert 3 < np.percentile(hps4, 25) < 10
    assert 20 < np.percentile(hps4, 50) < 40
    assert 100 < np.percentile(hps4, 75) < 250

    counter = Counter(hps5)
    assert len(counter) == 3

    assert 150 < counter['a'] < 250  # should be about 200
    assert 150 < counter['b'] < 250  # should be about 200
    assert 150 < counter['c'] < 250  # should be about 200


def _int_encode(val, lower, upper):
    denom = upper - lower + 1
    return (val - lower + 0.5) / denom


def test_get_ndarray_bounds():
    config_space = {
        '0': uniform(1.0, 1000.0),
        '1': loguniform(1.0, 1000.0),
        '2': reverseloguniform(0.9, 0.9999),
        '3': randint(1, 1000),
        '4': lograndint(1, 1000),
        '5': choice(['a', 'b', 'c'])}
    hp_ranges = make_hyperparameter_ranges(config_space)
    for epochs, val_last_pos in ((3, 1), (9, 3), (81, 81), (27, 1), (27, 9)):
        config_space_ext = ExtendedConfiguration(
            hp_ranges=hp_ranges, resource_attr_key='epoch',
            resource_attr_range=(1, epochs))
        hp_ranges_ext = config_space_ext.hp_ranges_ext
        hp_ranges_ext.value_for_last_pos = val_last_pos
        bounds = hp_ranges_ext.get_ndarray_bounds()
        val_enc = _int_encode(val_last_pos, lower=1, upper=epochs)
        assert all(x == (0.0, 1.0) for x in bounds[:-1])
        val_enc_cmp = bounds[-1][0]
        assert val_enc_cmp == bounds[-1][1]
        np.testing.assert_almost_equal(val_enc, val_enc_cmp, decimal=5)


def test_active_ranges_valid():
    config_space = {
        '0': uniform(1.0, 1000.0),
        '1': loguniform(1.0, 1000.0),
        '2': reverseloguniform(0.9, 0.9999),
        '3': randint(1, 1000),
        '4': lograndint(1, 1000),
        '5': choice(['a', 'b', 'c'])}
    invalid_active_spaces = [
        {
            '6': randint(0, 1),
        },
        {
            '0': uniform(2.0, 500.0),
            '5': choice(['a', 'b', 'd']),
        },
        {
            '0': uniform(2.0, 1000.0),
            '1': uniform(2.0, 500.0),
        },
        {
            '3': randint(1, 100),
            '4': lograndint(2, 1005),
        },
        {
            '2': reverseloguniform(0.99, 0.99999),
            '3': randint(5, 500),
        },
        {
            '2': reverseloguniform(0.9, 0.999),
            '4': lograndint(10, 1005),
        },
    ]
    for active_config_space in invalid_active_spaces:
        with pytest.raises(AssertionError):
            hp_ranges = HyperparameterRangesImpl(
                config_space=config_space,
                active_config_space=active_config_space)


@pytest.mark.parametrize('config_space,active_config_space', [
    ({
        '0': uniform(1.0, 2.0),
        '1': choice(['a', 'b', 'c']),
    },{
        '0': uniform(1.1, 1.9),
        '1': choice(['a', 'c']),
     }),
    ({
        '0': randint(1, 3),
        '1': choice(['a', 'c', 'b']),
    },{
        '0': randint(2, 3),
        '1': choice(['b', 'c']),
    }),
    ({
        '0': lograndint(3, 5),
        '1': randint(2, 3),
    },{
        '0': lograndint(3, 4),
    })
])
def test_active_ranges_samples(config_space, active_config_space):
    seed = 31415927
    random_state = np.random.RandomState(seed)
    hp_ranges = HyperparameterRangesImpl(
        config_space=config_space,
        active_config_space=active_config_space)
    configs = hp_ranges.random_configs(random_state, num_configs=100)
    _active_config_space = dict(config_space, **active_config_space)
    hp_ranges2 = HyperparameterRangesImpl(config_space=_active_config_space)
    # This fails with high probability if the sampled configs fall outside of
    # the narrower active ranges
    features = hp_ranges2.to_ndarray_matrix(configs)


def _cast_config(config, config_space):
    return {name: domain.cast(config[name])
            for name, domain in config_space.items()}


@pytest.mark.parametrize('config1,config2,match', [
    ({'1': 1}, {'1': 1}, True),
    ({'0': 0.546003}, {}, False),
    ({'1': 3}, {}, False),
    ({'2': 'b'}, {}, False),
    ({'3': 0.3}, {}, False),
    ({'4': 1}, {}, False),
    ({'5': 0.0001}, {}, False),
    ({'5': 0.0010005}, {}, True),
    ({'0': 0.546000000000001}, {}, True),
    ({'5': 0.01}, {'5': 0.01000001}, True),
])
def test_config_to_match_string(config1, config2, match):
    config_space = {
        '0': uniform(0.0, 1.0),
        '1': randint(1, 10),
        '2': choice(['a', 'b', 'c']),
        '3': finrange(0.1, 1.0, 10),
        '4': choice([3, 2, 1]),
        '5': choice([0.01, 0.001, 0.0001, 0.00001]),
    }
    hp_ranges = make_hyperparameter_ranges(config_space)

    config_base = {'0': 0.546, '1': 4, '2': 'a', '3': 0.4, '4': 3, '5': 0.001}
    _config1 = _cast_config(dict(config_base, **config1), config_space)
    _config2 = _cast_config(dict(config_base, **config2), config_space)
    match_str1 = hp_ranges.config_to_match_string(_config1)
    match_str2 = hp_ranges.config_to_match_string(_config2)
    assert (match_str1 == match_str2) == match, \
        f"match = {match}\nmatch_str1 = {match_str1}\nmatch_str2 = {match_str2}"


def test_config_space_for_sampling():
    config_space = {'0': uniform(0.0, 2.0)}
    active_space = {'0': uniform(0.5, 1.5)}
    hp_ranges = make_hyperparameter_ranges(
        config_space=config_space, active_config_space=active_space
    )
    assert hp_ranges.config_space_for_sampling == active_space, \
        "Active space should be used for sampling when specified."

    hp_ranges = make_hyperparameter_ranges(
        config_space=config_space
    )
    assert hp_ranges.config_space_for_sampling == config_space, \
        "Incorrect config space is used for sampling."


def test_encoded_ranges():
    config_space = {
        '0': uniform(0.0, 1.0),
        '1': randint(1, 10),
        '2': choice(['a', 'b', 'c']),
        '3': finrange(0.1, 1.0, 10),
        '4': choice([3, 2, 1]),
        '5': choice([0.01, 0.001, 0.0001, 0.00001]),
        '6': choice(['a', 'b'])
    }
    hp_ranges = make_hyperparameter_ranges(config_space)
    encoded_ranges = hp_ranges.encoded_ranges
    assert encoded_ranges['0'] == (0, 1)
    assert encoded_ranges['1'] == (1, 2)
    assert encoded_ranges['2'] == (2, 5)
    assert encoded_ranges['3'] == (5, 6)
    assert encoded_ranges['4'] == (6, 9)
    assert encoded_ranges['5'] == (9, 13)
    assert encoded_ranges['6'] == (13, 14)
