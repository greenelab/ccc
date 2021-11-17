"""
Tests other correlation coefficients, such as MIC and distance correlation.

PPS?
"""

import numpy as np
from sklearn.preprocessing import minmax_scale

from clustermatch.methods import mic, distcorr


def test_mine_random():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.random.rand(100)
    feature1 = np.random.rand(100)

    # Run
    val = mic(feature0, feature1)

    # Validate
    assert val is not None
    assert isinstance(val, float)
    assert 0.0 < np.round(val, 3) < 0.30


def test_mine_linear():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0

    # Run
    val = mic(feature0, feature1)

    # Validate
    assert np.round(val, 2) == 1.0


def test_mine_quadratic():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects with a linear relationship
    feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0)

    # Run
    val = mic(feature0, feature1)

    # Validate
    assert np.round(val, 2) == 1.0


def test_distcorr_random():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.random.rand(100)
    feature1 = np.random.rand(100)

    # Run
    val = distcorr(feature0, feature1)

    # Validate
    assert val is not None
    assert isinstance(val, float)
    assert 0.0 < np.round(val, 3) < 0.14


def test_distcorr_linear():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0

    # Run
    val = distcorr(feature0, feature1)

    # Validate
    assert np.round(val, 2) == 1.0


def test_distcorr_quadratic():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects with a linear relationship
    feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0)

    # Run
    val = distcorr(feature0, feature1)

    # Validate
    assert 0.50 < np.round(val, 2) < 0.53
