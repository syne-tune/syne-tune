# TODO: This code tests HyperparameterRanges and XYZScaling.
# If the latter code is removed, this test can go as well.

from collections import Counter
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from pytest import approx

from sagemaker_tune.search_space import uniform, randint, choice, loguniform, \
    lograndint
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from sagemaker_tune.optimizer.schedulers.searchers.bayesopt.datatypes.config_ext \
    import ExtendedConfiguration


def _assert_allclose_config(c1, c2, hp_ranges):
    c1_tpl = hp_ranges.config_to_tuple(c1)
    c2_tpl = hp_ranges.config_to_tuple(c2)
    assert_allclose(c1_tpl, c2_tpl)


@pytest.mark.parametrize('lower,upper,external_hp,internal_ndarray,domain', [
    (0.0, 8.0, 0.0, 0.0, uniform),
    (0.0, 8.0, 8.0, 1.0, uniform),
    (0.0, 8.0, 2.0, 0.25, uniform),
    (100.2, 100.6, 100.4, 0.5, uniform),
    (-2.0, 8.0, 0.0, 0.2, uniform),
    (-11.0, -1.0, -10.0, 0.1, uniform),
    (1.0, 8.0, 1.0, 0.0, loguniform),
    (1.0, 8.0, 8.0, 1.0, loguniform),
    (1.0, 10000.0, 10.0, 0.25, loguniform),
    (1.0, 10000.0, 100.0, 0.5, loguniform),
    (1.0, 10000.0, 1000.0, 0.75, loguniform),
    (0.001, 0.1, 0.01, 0.5, loguniform),
    (0.1, 100, 1.0, 1.0/3, loguniform),
    (1, 10001, 5001, 0.5, randint),
    (-10, 10, 0, 0.5, randint),
])
def test_continuous_to_and_from_ndarray(
        lower, upper, external_hp, internal_ndarray, domain):
    hp_ranges = make_hyperparameter_ranges({'a': domain(lower, upper)})
    config = hp_ranges.tuple_to_config((external_hp,))
    config_enc = np.array([internal_ndarray])
    assert_allclose(hp_ranges.to_ndarray(config), config_enc)
    _assert_allclose_config(
        hp_ranges.from_ndarray(config_enc), config, hp_ranges)


@pytest.mark.parametrize('choices,external_hp,internal_ndarray', [
    (['a', 'b'], 'a', [1.0, 0.0]),
    (['a', 'b'], 'b', [0.0, 1.0]),
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

    hps0, hps1, hps3, hps4, hps5 = zip(*[
        hp_ranges.config_to_tuple(x) for x in random_candidates])
    assert 200 < np.percentile(hps0, 25) < 300
    assert 450 < np.percentile(hps0, 50) < 550
    assert 700 < np.percentile(hps0, 75) < 800

    # same bounds as the previous but log scaling
    assert 3 < np.percentile(hps1, 25) < 10
    assert 20 < np.percentile(hps1, 50) < 40
    assert 100 < np.percentile(hps1, 75) < 200

    # integer
    assert 200 < np.percentile(hps3, 25) < 300
    assert 450 < np.percentile(hps3, 50) < 550
    assert 700 < np.percentile(hps3, 75) < 800

    # same bounds as the previous but log scaling
    assert 3 < np.percentile(hps4, 25) < 10
    assert 20 < np.percentile(hps4, 50) < 40
    assert 100 < np.percentile(hps4, 75) < 200

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
        '3': randint(1, 1000),
        '4': lograndint(1, 1000),
        '5': choice(['a', 'b', 'c'])}
    hp_ranges = make_hyperparameter_ranges(config_space)
    for epochs, val_last_pos in ((3, 1), (9, 3), (81, 81), (27, 1), (27, 9)):
        configspace_ext = ExtendedConfiguration(
            hp_ranges=hp_ranges, resource_attr_key='epoch',
            resource_attr_range=(1, epochs))
        hp_ranges_ext = configspace_ext.hp_ranges_ext
        hp_ranges_ext.value_for_last_pos = val_last_pos
        bounds = hp_ranges_ext.get_ndarray_bounds()
        val_enc = _int_encode(val_last_pos, lower=1, upper=epochs)
        assert all(x == (0.0, 1.0) for x in bounds[:-1])
        val_enc_cmp = bounds[-1][0]
        assert val_enc_cmp == bounds[-1][1]
        np.testing.assert_almost_equal(val_enc, val_enc_cmp, decimal=5)
