import numpy as np

from ccc.methods import mic


def test_mic_basic():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.random.rand(100)
    feature1 = np.random.rand(100)

    # Run
    mic_value = mic(feature0, feature1)
    assert mic_value is not None
    assert isinstance(mic_value, float)
    assert 1.0 > mic_value > 0.0


def test_mic_use_estimator_mic_e():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.random.rand(100)
    feature1 = np.random.rand(100)

    # Run default estimator
    mic_value = mic(feature0, feature1)

    # Run with mic_e estimator
    mic_e_value = mic(feature0, feature1, estimator="mic_e")

    assert mic_e_value is not None
    assert isinstance(mic_e_value, float)
    assert 1.0 > mic_e_value > 0.0

    # make sure the estimator parameter is being used
    assert mic_value != mic_e_value
