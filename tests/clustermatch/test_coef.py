import numpy as np
from sklearn.preprocessing import minmax_scale

from clustermatch.coef import cm


def test_cm_basic():
    ## Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.random.rand(100)
    feature1 = np.random.rand(100)

    ## Run
    cm_value = cm(feature0, feature1)

    ## Validate
    assert cm_value is not None
    assert isinstance(cm_value, float)


def test_cm_random_data():
    ## Prepare
    np.random.seed(123)

    for i in range(10):
        # two features on 100 objects (random data)
        feature0 = minmax_scale(
            np.random.rand(100), (-1.0, 1.0)
        )  # with negative values
        feature1 = np.random.rand(100)  # all positive values between 0 and 1

        ## Run
        cm_value = cm(feature0, feature1)

        ## Validate
        assert cm_value < 0.05


def test_cm_linear():
    ## Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0

    ## Run
    cm_value = cm(feature0, feature1)

    ## Validate
    assert cm_value == 1.0


def test_cm_quadratic():
    ## Prepare
    np.random.seed(1)

    # two features on 100 objects with a linear relationship
    feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0)

    ## Run
    cm_value = cm(feature0, feature1)

    ## Validate
    assert cm_value > 0.40


def test_cm_quadratic2():
    ## Prepare
    np.random.seed(1)

    # two features on 100 objects with a linear relationship
    feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0) + (0.10 * np.random.rand(feature0.shape[0]))

    ## Run
    cm_value = cm(feature0, feature1)

    ## Validate
    assert cm_value > 0.40


def test_feature_with_all_same_values():
    ## Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = np.array([5] * feature0.shape[0])

    ## Run
    cm_value = cm(feature0, feature1)

    ## Validate
    assert cm_value == 0.0


def test_all_features_with_all_same_values():
    ## Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.array([0] * 100)
    feature1 = np.array([5] * feature0.shape[0])

    ## Run
    cm_value = cm(feature0, feature1)

    ## Validate
    assert cm_value == 0.0


# TODO: _get_range_n_clusters !!! it's returning less clusters in range(2, 10) -> 8 clusters instead of 9

# TODO: test data has two features with different shape

# TODO: test data with some nan in feature0
# TODO: test data with some nan in feature1
# TODO: test data with all nan in feature0
# TODO: test data with all nan in feature1

# TODO: add stats options to get the partitions or number of clusters that
#  generated each cm value (this will be useful to debug the method as we
#  talked with Diego)
