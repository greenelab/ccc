from random import shuffle

import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import adjusted_rand_score as ari

from clustermatch.coef import (
    cm,
    _get_range_n_clusters,
    run_quantile_clustering,
    _get_perc_from_k,
    _get_parts,
)


def test_get_perc_from_k_with_k_less_than_two():
    assert _get_perc_from_k(1) == []
    assert _get_perc_from_k(0) == []
    assert _get_perc_from_k(-1) == []


def test_get_perc_from_k():
    assert _get_perc_from_k(2) == [0.5]
    assert np.round(_get_perc_from_k(3), 3).tolist() == [0.333, 0.667]
    assert _get_perc_from_k(4) == [0.25, 0.50, 0.75]


def test_run_quantile_clustering_with_two_clusters01():
    # Prepare
    np.random.seed(0)

    data = np.concatenate(
        (
            np.random.normal(0, 1, 10),
            np.random.normal(5, 1, 10),
        )
    )
    data_ref = np.concatenate(([0] * 10, [1] * 10))

    idx_shuffled = list(range(len(data)))
    shuffle(idx_shuffled)

    data = data[idx_shuffled]
    data_ref = data_ref[idx_shuffled]

    # Run
    part = run_quantile_clustering(data, 2)

    # Validate
    assert part is not None
    assert len(part) == 20
    assert len(np.unique(part)) == 2
    assert ari(data_ref, part) == 1.0, ari(data_ref, part)


def test_run_quantile_clustering_with_two_clusters_mixed():
    # Prepare
    np.random.seed(0)

    data = np.concatenate(
        (
            np.random.normal(-3, 0.5, 5),
            np.random.normal(0, 1, 5),
            np.random.normal(5, 1, 5),
            np.random.normal(10, 1, 5),
        )
    )
    data_ref = np.concatenate(([0] * 10, [1] * 10))

    idx_shuffled = list(range(len(data)))
    shuffle(idx_shuffled)

    data = data[idx_shuffled]
    data_ref = data_ref[idx_shuffled]

    # Run
    part = run_quantile_clustering(data, 2)

    # Validate
    assert part is not None
    assert len(part) == 20
    assert len(np.unique(part)) == 2
    assert ari(data_ref, part) == 1.0, ari(data_ref, part)


def test_run_quantile_clustering_with_four_clusters():
    # Prepare
    np.random.seed(0)

    data = np.concatenate(
        (
            np.random.normal(-3, 0.5, 5),
            np.random.normal(0, 1, 5),
            np.random.normal(5, 1, 5),
            np.random.normal(10, 1, 5),
        )
    )
    data_ref = np.concatenate(([0] * 5, [1] * 5, [2] * 5, [3] * 5))

    idx_shuffled = list(range(len(data)))
    shuffle(idx_shuffled)

    data = data[idx_shuffled]
    data_ref = data_ref[idx_shuffled]

    # Run
    part = run_quantile_clustering(data, 4)

    # Validate
    assert part is not None
    assert len(part) == 20
    assert len(np.unique(part)) == 4
    assert ari(data_ref, part) == 1.0


def test_get_range_n_clusters_without_internal_n_clusters():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100)
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4, 5, 6, 7, 8, 9, 10)

    # 25 features
    range_n_clusters = _get_range_n_clusters(25)
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4, 5)


def test_get_range_n_clusters_with_internal_n_clusters_is_list():
    # 100 features
    range_n_clusters = _get_range_n_clusters(
        100,
        internal_n_clusters=[
            2,
        ],
    )
    assert range_n_clusters is not None
    assert range_n_clusters == (2,)

    # 25 features
    range_n_clusters = _get_range_n_clusters(
        25,
        internal_n_clusters=[
            2,
        ],
    )
    assert range_n_clusters is not None
    assert range_n_clusters == (2,)

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=[2, 3, 4])
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4)


def test_get_range_n_clusters_with_internal_n_clusters_empty():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[])
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4, 5, 6, 7, 8, 9, 10)

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=[])
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4, 5)


def test_get_range_n_clusters_with_internal_n_clusters_is_tuple():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=(2,))
    assert range_n_clusters is not None
    assert range_n_clusters == (2,)

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=(2,))
    assert range_n_clusters is not None
    assert range_n_clusters == (2,)

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=(2, 3, 4))
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4)


def test_get_range_n_clusters_with_internal_n_clusters_is_range():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=range(2, 3))
    assert range_n_clusters is not None
    assert range_n_clusters == (2,)

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=range(2, 3))
    assert range_n_clusters is not None
    assert range_n_clusters == (2,)

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=range(2, 4 + 1))
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4)


def test_get_range_n_clusters_with_internal_n_clusters_is_int():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=2)
    assert range_n_clusters is not None
    assert range_n_clusters == (2,)

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=3)
    assert range_n_clusters is not None
    assert range_n_clusters == (3,)

    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=1)
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4, 5)


def test_get_range_n_clusters_with_internal_n_clusters_are_less_than_two():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 3, 4])
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4)

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 1, 4])
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 4)

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 3, 1])
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3)

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 0, 4])
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 4)

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 1, -4, 6])
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 6)


def test_get_range_n_clusters_with_internal_n_clusters_are_repeated():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[2, 3, 2, 4])
    assert range_n_clusters is not None
    assert range_n_clusters == (2, 3, 4)

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[2, 2, 2])
    assert range_n_clusters is not None
    assert range_n_clusters == (2,)


def test_get_range_n_clusters_with_very_few_features():
    # 3 features
    range_n_clusters = _get_range_n_clusters(3)
    assert range_n_clusters is not None
    assert range_n_clusters == (2,)

    # 2 features
    range_n_clusters = _get_range_n_clusters(2)
    assert range_n_clusters is not None
    assert range_n_clusters == tuple()

    # 1 features
    range_n_clusters = _get_range_n_clusters(1)
    assert range_n_clusters is not None
    assert range_n_clusters == tuple()

    # 0 features
    range_n_clusters = _get_range_n_clusters(0)
    assert range_n_clusters is not None
    assert range_n_clusters == tuple()


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


def test_cm_feature_with_all_same_values():
    ## Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = np.array([5] * feature0.shape[0])

    ## Run
    cm_value = cm(feature0, feature1)

    ## Validate
    assert np.isnan(cm_value)


def test_cm_all_features_with_all_same_values():
    # this test generates internal partitions with only one cluster. In this case,
    # clustermatch returns NaN

    ## Prepare
    np.random.seed(0)

    # two features with constant values
    feature0 = np.array([0] * 100)
    feature1 = np.array([5] * feature0.shape[0])

    ## Run
    cm_value = cm(feature0, feature1)

    ## Validate
    assert np.isnan(cm_value)


def test_cm_single_argument_is_matrix():
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0
    feature2 = np.random.rand(feature0.shape[0])

    input_data = np.array([feature0, feature1, feature2])

    # Run
    cm_value = cm(input_data)

    # Validate
    assert cm_value is not None
    assert hasattr(cm_value, "shape")
    assert cm_value.shape == (3,)
    # assert np.array_equal(np.diag(cm_value), np.ones(cm_value.shape[0]))

    assert cm_value[0] == 1.0
    assert cm_value[1] < 0.03
    assert cm_value[2] < 0.03


def test_cm_single_argument_is_matrix_with_precomputing_of_parts():
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0
    feature2 = np.random.rand(feature0.shape[0])

    input_data = np.array([feature0, feature1, feature2])

    # Run
    cm_value = cm(input_data, precompute_parts=True)

    # Validate
    assert cm_value is not None
    assert hasattr(cm_value, "shape")
    assert cm_value.shape == (3,)
    # assert np.array_equal(np.diag(cm_value), np.ones(cm_value.shape[0]))

    assert cm_value[0] == 1.0
    assert cm_value[1] < 0.03
    assert cm_value[2] < 0.03


# TODO: test data has two features with different shape

# TODO: test data with some nan in feature0
# TODO: test data with some nan in feature1
# TODO: test data with all nan in feature0
# TODO: test data with all nan in feature1

# TODO: add stats options to get the partitions or number of clusters that
#  generated each cm value (this will be useful to debug the method as we
#  talked with Diego)


def test_get_parts_simple():
    np.random.seed(0)

    feature0 = np.random.rand(100)

    # run
    parts = _get_parts(feature0, (2,))
    assert parts is not None
    assert len(parts) == 1
    assert len(np.unique(parts[0])) == 2

    parts = _get_parts(feature0, (2, 3))
    assert parts is not None
    assert len(parts) == 2
    assert len(np.unique(parts[0])) == 2
    assert len(np.unique(parts[1])) == 3


def test_get_parts_k_is_greater_or_equal_than_n_objects():
    np.random.seed(0)

    feature0 = np.random.rand(10)

    # run
    parts = _get_parts(feature0, (10,))
    assert parts is not None
    assert len(parts) == 0

    parts = _get_parts(feature0, (20,))
    assert parts is not None
    assert len(parts) == 0
