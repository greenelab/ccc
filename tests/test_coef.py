from concurrent.futures import ThreadPoolExecutor
from random import shuffle
from unittest.mock import patch
import time
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import adjusted_rand_score as ari

from ccc.coef import (
    ccc,
    get_range_n_clusters,
    run_quantile_clustering,
    get_perc_from_k,
    get_parts,
    get_coords_from_index,
    cdist_parts_basic,
    cdist_parts_parallel,
    get_chunks,
    get_n_workers,
)


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_get_perc_from_k_with_k_less_than_two():
    assert get_perc_from_k(1) == []
    assert get_perc_from_k(0) == []
    assert get_perc_from_k(-1) == []


def test_get_perc_from_k():
    assert get_perc_from_k(2) == [0.5]
    assert np.round(get_perc_from_k(3), 3).tolist() == [0.333, 0.667]
    assert get_perc_from_k(4) == [0.25, 0.50, 0.75]


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
    range_n_clusters = get_range_n_clusters(100)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(
        range_n_clusters, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    )

    # 25 features
    range_n_clusters = get_range_n_clusters(25)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4, 5]))


def test_get_range_n_clusters_with_internal_n_clusters_is_list():
    # 100 features
    range_n_clusters = get_range_n_clusters(
        100,
        internal_n_clusters=[2],
    )
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))

    # 25 features
    range_n_clusters = get_range_n_clusters(
        25,
        internal_n_clusters=[2],
    )
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))

    # 25 features
    range_n_clusters = get_range_n_clusters(25, internal_n_clusters=[2, 3, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4]))


def test_get_range_n_clusters_with_internal_n_clusters_none():
    # 100 features
    range_n_clusters = get_range_n_clusters(100, internal_n_clusters=None)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(
        range_n_clusters, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    )

    # 25 features
    range_n_clusters = get_range_n_clusters(25, internal_n_clusters=None)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4, 5]))


def test_get_range_n_clusters_with_internal_n_clusters_has_single_int():
    # 100 features
    range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[2])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))

    # 25 features
    range_n_clusters = get_range_n_clusters(25, internal_n_clusters=[3])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([3]))

    # 5 features
    range_n_clusters = get_range_n_clusters(5, internal_n_clusters=[4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([4]))

    # 25 features but invalid number of clusters
    range_n_clusters = get_range_n_clusters(25, internal_n_clusters=[1])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))

    # 25 features but invalid number of clusters
    range_n_clusters = get_range_n_clusters(25, internal_n_clusters=[25])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))


def test_get_range_n_clusters_with_internal_n_clusters_are_less_than_two():
    # 100 features
    range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 3, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4]))

    range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 1, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 4]))

    range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 3, 1])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3]))

    range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 0, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 4]))

    range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[1, 2, 1, -4, 6])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 6]))


def test_get_range_n_clusters_with_internal_n_clusters_are_repeated():
    # 100 features
    range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[2, 3, 2, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4]))

    range_n_clusters = get_range_n_clusters(100, internal_n_clusters=[2, 2, 2])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))


def test_get_range_n_clusters_with_very_few_features():
    # 3 features
    range_n_clusters = get_range_n_clusters(3)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))

    # 2 features
    range_n_clusters = get_range_n_clusters(2)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))

    # 1 features
    range_n_clusters = get_range_n_clusters(1)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))

    # 0 features
    range_n_clusters = get_range_n_clusters(0)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))


def test_get_range_n_clusters_with_larger_k_than_features():
    # 10 features
    range_n_clusters = get_range_n_clusters(10, internal_n_clusters=[10])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))

    # 10 features
    range_n_clusters = get_range_n_clusters(10, internal_n_clusters=[11])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))


def test_get_range_n_clusters_with_default_max_k():
    range_n_clusters = get_range_n_clusters(200)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(
        range_n_clusters, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    )


def test_cm_basic():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.random.rand(100)
    feature1 = np.random.rand(100)

    # Run
    cm_value = ccc(feature0, feature1)
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.01, abs=0.01)


def test_cm_basic_internal_n_clusters_is_integer():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.random.rand(100)
    feature1 = np.random.rand(100)

    # Run
    cm_value = ccc(feature0, feature1)
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value > 0.0

    # Run with internal_n_clusters equals to default value but as integer
    cm_value2 = ccc(feature0, feature1, internal_n_clusters=10)
    assert cm_value == cm_value2


def test_cm_basic_internal_n_clusters_is_integer_more_checks():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.random.rand(100)
    feature1 = np.random.rand(100)

    # Run
    cm_value = ccc(feature0, feature1, internal_n_clusters=[2, 3, 4])
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value > 0.0

    # Run with internal_n_clusters equals to default value but as integer
    cm_value2 = ccc(feature0, feature1, internal_n_clusters=4)
    assert cm_value == cm_value2


def test_cm_ari_is_negative():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.array([1, 2, 3, 4, 5])
    feature1 = np.array([2, 4, 1, 3, 5])

    # Run
    cm_value = ccc(feature0, feature1)

    # Validate
    # ari for this example is -0.25, but cm should return 0.0
    assert cm_value == 0.0


def test_cm_random_data():
    # Prepare
    rs = np.random.RandomState(123)

    for i in range(10):
        # two features on 100 objects (random data)
        feature0 = minmax_scale(rs.rand(100), (-1.0, 1.0))  # with negative values
        feature1 = rs.rand(100)  # all positive values between 0 and 1

        # Run
        cm_value = ccc(feature0, feature1)

        # Validate
        assert cm_value == pytest.approx(0.025, abs=0.025)


def test_cm_linear():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0

    # Run
    cm_value = ccc(feature0, feature1)

    # Validate
    assert cm_value == 1.0


def test_cm_quadratic():
    # Prepare
    np.random.seed(1)

    # two features on 100 objects with a quadratic relationship
    feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0)

    # Run
    cm_value = ccc(feature0, feature1)

    # Validate
    assert cm_value > 0.40


def test_cm_quadratic_noisy():
    # Prepare
    np.random.seed(1)

    # two features on 100 objects with a quadratic relationship
    feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0) + (0.10 * np.random.rand(feature0.shape[0]))

    # Run
    cm_value = ccc(feature0, feature1)

    # Validate
    assert cm_value > 0.40


def test_cm_one_feature_with_all_same_values():
    # if there is no variation in at least one of the two variables to be
    #  compared, ccc returns nan

    # Prepare
    np.random.seed(0)

    # two features on 100 objects; all values in feature1 are the same
    feature0 = np.random.rand(100)
    feature1 = np.array([5] * feature0.shape[0])

    # Run
    cm_value = ccc(feature0, feature1)

    # Validate
    assert np.isnan(cm_value), cm_value


def test_cm_all_features_with_all_same_values():
    # if there is no variation in both variables to be compared, ccc
    #  returns nan

    # Prepare
    np.random.seed(0)

    # two features with constant values
    feature0 = np.array([0] * 100)
    feature1 = np.array([5] * feature0.shape[0])

    # Run
    cm_value = ccc(feature0, feature1)

    # Validate
    assert np.isnan(cm_value)


def test_cm_single_argument_is_matrix():
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0
    feature2 = np.random.rand(feature0.shape[0])

    input_data = np.array([feature0, feature1, feature2])

    # Run
    cm_value = ccc(input_data)

    # Validate
    assert cm_value is not None
    assert hasattr(cm_value, "shape")
    assert cm_value.shape == (3,)
    # assert np.array_equal(np.diag(cm_value), np.ones(cm_value.shape[0]))

    assert cm_value[0] == 1.0
    assert cm_value[1] < 0.03
    assert cm_value[2] < 0.03


def test_cm_x_y_are_pandas_series():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = pd.Series(np.random.rand(100))
    feature1 = pd.Series(np.random.rand(100))

    # Run
    cm_value = ccc(feature0, feature1)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, float)


def test_cm_x_and_y_are_pandas_dataframe():
    # two arguments are dataframes (invalid)
    x = pd.DataFrame(np.random.rand(10, 100))
    y = pd.DataFrame(np.random.rand(10, 100))

    # Run
    with pytest.raises(ValueError) as e:
        ccc(x, y)

    assert "wrong combination" in str(e).lower()


def test_cm_integer_overflow_random():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(1000000)
    feature1 = np.random.rand(1000000)

    # Run
    cm_value = ccc(feature0, feature1)
    assert 0.0 <= cm_value <= 0.01


def test_cm_integer_overflow_perfect_match():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(1000000)

    # Run
    cm_value = ccc(feature0, feature0)
    assert cm_value == 1.0


def test_get_parts_simple():
    np.random.seed(0)

    feature0 = np.random.rand(100)

    # run
    parts = get_parts(feature0, (2,))
    assert parts is not None
    assert len(parts) == 1
    assert len(np.unique(parts[0])) == 2

    parts = get_parts(feature0, (2, 3))
    assert parts is not None
    assert len(parts) == 2
    assert len(np.unique(parts[0])) == 2
    assert len(np.unique(parts[1])) == 3


def test_get_parts_with_singletons():
    np.random.seed(0)

    feature0 = np.array([1.3] * 10)

    # run
    parts = get_parts(feature0, (2,))
    assert parts is not None
    assert len(parts) == 1
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([-2]))

    parts = get_parts(feature0, (2, 3))
    assert parts is not None
    assert len(parts) == 2
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([-2]))
    np.testing.assert_array_equal(np.unique(parts[1]), np.array([-2]))


def test_get_parts_with_categorical_feature():
    np.random.seed(0)

    feature0 = np.array([4] * 10)

    # run
    # only one partition is requested
    parts = get_parts(feature0, (2,), data_is_numerical=False)
    assert parts is not None
    assert len(parts) == 1
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([4]))

    # more partitions are requested; only the first one has valid information
    parts = get_parts(feature0, (2, 3), data_is_numerical=False)
    assert parts is not None
    assert len(parts) == 2
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([4]))
    np.testing.assert_array_equal(np.unique(parts[1]), np.array([-1]))


def test_cdist_parts_one_vs_one():
    from scipy.spatial.distance import cdist
    from sklearn.metrics import adjusted_rand_score as ari

    parts0 = np.array(
        [
            [1, 1, 2, 2, 3, 3],
        ]
    )
    parts1 = np.array(
        [
            [3, 3, 1, 1, 2, 2],
        ]
    )

    expected_cdist = cdist(parts0, parts1, metric=ari)
    np.testing.assert_array_equal(expected_cdist, np.array([[1.0]]))

    # basic version (one thread)
    observed_cdist = cdist_parts_basic(parts0, parts1)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)

    # with one thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)

    # with two threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)


def test_cdist_parts_one_vs_one_dissimilar():
    from scipy.spatial.distance import cdist
    from sklearn.metrics import adjusted_rand_score as ari

    parts0 = np.array(
        [
            [1, 1, 2, 1, 3, 3],
        ]
    )
    parts1 = np.array(
        [
            [3, 3, 1, 1, 2, 3],
        ]
    )

    expected_cdist = cdist(parts0, parts1, metric=ari)
    np.testing.assert_array_equal(expected_cdist, np.array([[-0.022727272727272728]]))

    # basic version (one thread)
    observed_cdist = cdist_parts_basic(parts0, parts1)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)

    # with one thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)

    # with two threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)


def test_cdist_parts_one_vs_two():
    from scipy.spatial.distance import cdist
    from sklearn.metrics import adjusted_rand_score as ari

    parts0 = np.array(
        [
            [1, 1, 2, 1, 3, 3],
        ]
    )
    parts1 = np.array(
        [
            [3, 3, 1, 1, 2, 3],
            [3, 3, 1, 1, 2, 2],
        ]
    )

    expected_cdist = cdist(parts0, parts1, metric=ari)
    np.testing.assert_array_equal(
        expected_cdist,
        np.array(
            [
                [-0.022727272727272728, 0.4444444444444444],
            ]
        ),
    )

    # basic version (one thread)
    observed_cdist = cdist_parts_basic(parts0, parts1)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)

    # with one thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)

    # with two threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)


def test_cdist_parts_two_vs_two():
    from scipy.spatial.distance import cdist
    from sklearn.metrics import adjusted_rand_score as ari

    parts0 = np.array(
        [
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 1, 3, 3],
        ]
    )
    parts1 = np.array(
        [
            [3, 3, 1, 1, 2, 3],
            [3, 3, 1, 1, 2, 2],
        ]
    )

    expected_cdist = cdist(parts0, parts1, metric=ari)
    np.testing.assert_array_equal(
        expected_cdist,
        np.array(
            [
                [0.4444444444444444, 1.0],
                [-0.022727272727272728, 0.4444444444444444],
            ]
        ),
    )

    # basic version (one thread)
    observed_cdist = cdist_parts_basic(parts0, parts1)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)

    # with one thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)

    # with two threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        observed_cdist = cdist_parts_parallel(parts0, parts1, executor)
    np.testing.assert_array_equal(observed_cdist, expected_cdist)


def test_get_coords_from_index():
    # data is an example with n_obj = 5 just to illustrate
    # data = np.array(
    #     [
    #         [10, 11],
    #         [23, 22],
    #         [27, 26],
    #         [37, 36],
    #         [47, 46],
    #     ]
    # )
    n_obj = 5

    # index: 0 -> (0, 1) (row_idx, col_idx)
    res = get_coords_from_index(n_obj, 0)
    assert res == (0, 1)

    res = get_coords_from_index(n_obj, 1)
    assert res == (0, 2)

    res = get_coords_from_index(n_obj, 3)
    assert res == (0, 4)

    res = get_coords_from_index(n_obj, 4)
    assert res == (1, 2)

    # index: 9, the last one
    res = get_coords_from_index(n_obj, 9)
    assert res == (3, 4)


def test_get_coords_from_index_smaller():
    # data is an example with n_obj = 5 just to illustrate
    # data = np.array(
    #     [
    #         [10, 11],
    #         [23, 22],
    #         [27, 26],
    #         [37, 36],
    #     ]
    # )
    n_obj = 4

    # index: 0 -> (0, 1) (row_idx, col_idx)
    res = get_coords_from_index(n_obj, 0)
    assert res == (0, 1)

    res = get_coords_from_index(n_obj, 1)
    assert res == (0, 2)

    res = get_coords_from_index(n_obj, 2)
    assert res == (0, 3)

    res = get_coords_from_index(n_obj, 3)
    assert res == (1, 2)

    # index: 5, the last one
    res = get_coords_from_index(n_obj, 5)
    assert res == (2, 3)


def test_cm_values_equal_to_original_implementation():
    # compare with results obtained from the original ccc
    # implementation (https://github.com/sinc-lab/clustermatch) plus some
    # patches (see tests/data/README.md about ccc data).
    from pathlib import Path
    import pandas as pd

    # from pandas.testing import assert_frame_equal

    input_data_dir = Path(__file__).parent / "data"

    # load data
    data = pd.read_pickle(input_data_dir / "ccc-random_data-data.pkl")
    data = data.to_numpy()

    # run new ccc implementation.
    # Here, I fixed the internal number of clusters, since that slightly changed
    # in the new implementation compared with the original one.
    corr_mat = ccc(data, internal_n_clusters=list(range(2, 10 + 1)))

    expected_corr_matrix = pd.read_pickle(input_data_dir / "ccc-random_data-coef.pkl")
    expected_corr_matrix = expected_corr_matrix.to_numpy()
    expected_corr_matrix = expected_corr_matrix[
        np.triu_indices(expected_corr_matrix.shape[0], 1)
    ]

    np.testing.assert_almost_equal(
        expected_corr_matrix,
        corr_mat,
    )


def test_cm_return_parts_quadratic():
    # Prepare
    np.random.seed(0)

    # two features with a quadratic relationship
    feature0 = np.array([-4, -3, -2, -1, 0, 0, 1, 2, 3, 4])
    feature1 = np.array([10, 9, 8, 7, 6, 6, 7, 8, 9, 10])

    # Run
    cm_value, max_parts, parts = ccc(
        feature0, feature1, internal_n_clusters=[2, 3], return_parts=True
    )

    # Validate
    assert cm_value.round(2) == 0.31

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].shape == (2, 10)
    assert len(np.unique(parts[0][0])) == 2
    assert len(np.unique(parts[0][1])) == 3
    assert parts[1].shape == (2, 10)
    assert len(np.unique(parts[1][0])) == 2
    assert len(np.unique(parts[1][1])) == 3

    assert max_parts is not None
    assert hasattr(max_parts, "shape")
    assert max_parts.shape == (2,)
    # the set of partitions that maximize ari is:
    #   - k == 3 for feature0
    #   - k == 2 for feature1
    np.testing.assert_array_equal(max_parts, np.array([1, 0]))


def test_cm_return_parts_linear():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0

    # Run
    cm_value, max_parts, parts = ccc(feature0, feature1, return_parts=True)

    # Validate
    assert cm_value == 1.0

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].shape == (9, 100)
    assert parts[1].shape == (9, 100)

    assert max_parts is not None
    assert hasattr(max_parts, "shape")
    assert max_parts.shape == (2,)
    # even in this test we do not specify internal_n_clusters (so it goes from
    # k=2 to k=10, nine partitions), k=2 for both features should already have
    # the maximum value
    np.testing.assert_array_equal(max_parts, np.array([0, 0]))


def test_cm_return_parts_categorical_variable():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects
    numerical_feature0 = np.random.rand(100)
    numerical_feature0_median = np.percentile(numerical_feature0, 50)

    # create a categorical variable perfectly correlated with the numerical one (this is actually an ordinal feature)
    categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.str_)
    categorical_feature1[numerical_feature0 < numerical_feature0_median] = "l"
    categorical_feature1[numerical_feature0 >= numerical_feature0_median] = "u"
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 2
    assert set(_unique_values) == {"l", "u"}

    # Run
    cm_value, max_parts, parts = ccc(
        numerical_feature0, categorical_feature1, return_parts=True
    )

    # Validate
    assert cm_value == 1.0

    assert parts is not None
    assert len(parts) == 2

    # for numerical_feature0 all partititions should be there
    assert parts[0].shape == (9, 100)
    assert set(range(2, 10 + 1)) == set(map(lambda x: np.unique(x).shape[0], parts[0]))

    # for categorical_feature1 only the first partition is meaningful
    assert parts[1].shape == (9, 100)
    assert np.unique(parts[1][0, :]).shape[0] == 2
    _unique_in_rest = np.unique(parts[1][1:, :])
    assert _unique_in_rest.shape[0] == 1
    assert _unique_in_rest[0] == -1

    assert max_parts is not None
    assert hasattr(max_parts, "shape")
    assert max_parts.shape == (2,)
    # even in this test we do not specify internal_n_clusters (so it goes from
    # k=2 to k=10, nine partitions), k=2 for both features should already have
    # the maximum value
    np.testing.assert_array_equal(max_parts, np.array([0, 0]))


def test_cm_return_parts_with_matrix_as_input():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0
    X = pd.DataFrame(
        {
            "feature0": feature0,
            "feature1": feature1,
        }
    )

    # Run
    cm_value, max_parts, parts = ccc(X, return_parts=True)

    # Validate
    assert cm_value == 1.0

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].shape == (9, 100)
    assert parts[1].shape == (9, 100)

    assert max_parts is not None
    assert hasattr(max_parts, "shape")
    assert max_parts.shape == (2,)
    # even in this test we do not specify internal_n_clusters (so it goes from
    # k=2 to k=10, nine partitions), k=2 for both features should already have
    # the maximum value (because the relationship is linear)
    np.testing.assert_array_equal(max_parts, np.array([0, 0]))


def test_get_chunks_n_large():
    # n = 100 (even)
    n_comp = 100
    n_threads = 2

    cs = get_chunks(n_comp, n_threads)

    assert len(cs) == 2
    assert set(map(len, cs)) == {50}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n = 100 (even)
    n_comp = 100
    n_threads = 5

    cs = get_chunks(n_comp, n_threads)

    assert len(cs) == 5
    assert set(map(len, cs)) == {20}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n = 100 (even) not equal groups
    n_comp = 100
    n_threads = 8

    cs = get_chunks(n_comp, n_threads)

    assert len(cs) == 8
    assert set(map(len, cs)) == {9, 13}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n = 101 (odd)
    n_comp = 101
    n_threads = 2

    cs = get_chunks(n_comp, n_threads)

    assert len(cs) == 2
    assert set(map(len, cs)) == {50, 51}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))


def test_get_chunks_n_small():
    # n = 10
    n_comp = 10
    n_threads = 2

    cs = get_chunks(n_comp, n_threads)

    # for this cases, an automatic ratio will avoid singleton chunks, and
    # instead will generate n_threads groups
    assert len(cs) == 2
    assert set(map(len, cs)) == {5}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n = 10
    n_comp = 10
    n_threads = 5

    cs = get_chunks(n_comp, n_threads)

    assert len(cs) == 5
    assert set(map(len, cs)) == {2}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n = 11
    n_comp = 11
    n_threads = 2

    cs = get_chunks(n_comp, n_threads)

    # for this case, it will avoid singleton chunks
    assert len(cs) == 2
    assert set(map(len, cs)) == {5, 6}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n = 10 and n_threads is odd
    n_comp = 10
    n_threads = 3

    cs = get_chunks(n_comp, n_threads)

    assert len(cs) == 3
    assert set(map(len, cs)) == {4, 2}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))


def test_get_chunks_n_small_n_threads_large():
    # in all these cases, n_thread is larger than n_comp / 2, but always
    # smaller than n_comp. The problem in this test is that we want to make sure
    # that we are using all threads

    # n_threads = 6
    n_comp = 10
    n_threads = 6
    cs = get_chunks(n_comp, n_threads)
    assert len(cs) == 6
    assert set(map(len, cs)) == {1, 2}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n_threads = 7
    n_comp = 10
    n_threads = 7
    cs = get_chunks(n_comp, n_threads)
    assert len(cs) == 7
    assert set(map(len, cs)) == {1, 2}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n_threads = 8
    n_comp = 10
    n_threads = 8
    cs = get_chunks(n_comp, n_threads)
    assert len(cs) == 8
    assert set(map(len, cs)) == {1, 2}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n_threads = 9
    n_comp = 10
    n_threads = 9
    cs = get_chunks(n_comp, n_threads)
    assert len(cs) == 9
    assert set(map(len, cs)) == {1, 2}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))


def test_get_chunks_n_small_n_threads_very_large():
    # in all these cases, n_thread is at least as large as n_comp.
    # that we are using all threads
    # n_threads = 10
    n_comp = 10
    n_threads = 10
    cs = get_chunks(n_comp, n_threads)
    assert len(cs) == 10
    assert set(map(len, cs)) == {1}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # n_threads = 11
    n_comp = 10
    n_threads = 11
    cs = get_chunks(n_comp, n_threads)
    assert len(cs) == 10
    assert set(map(len, cs)) == {1}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))


def test_get_chunks_ratio_is_one():
    n_comp = 10
    n_threads = 2
    ratio = 1

    cs = get_chunks(n_comp, n_threads, ratio)

    assert len(cs) == 2
    assert set(map(len, cs)) == {5}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))


def test_get_chunks_ratio_is_two():
    n_comp = 10
    n_threads = 2
    ratio = 2

    cs = get_chunks(n_comp, n_threads, ratio)

    assert len(cs) == 4
    assert set(map(len, cs)) == {1, 3}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))


def test_get_chunks_ratio_is_large():
    # ratio = 4
    n_comp = 10
    n_threads = 2
    ratio = 4

    cs = get_chunks(n_comp, n_threads, ratio)

    assert len(cs) == 8
    assert set(map(len, cs)) == {1, 2}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # ratio = 5
    n_comp = 10
    n_threads = 2
    ratio = 5

    cs = get_chunks(n_comp, n_threads, ratio)

    assert len(cs) == 10
    assert set(map(len, cs)) == {1}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))

    # ratio = 6
    n_comp = 10
    n_threads = 2
    ratio = 6

    cs = get_chunks(n_comp, n_threads, ratio)

    assert len(cs) == 10
    assert set(map(len, cs)) == {1}
    assert {x for i in cs for x in i} == set(np.arange(n_comp))


def test_get_chunks_iterable():
    # here the first argument is an iterable, not an integer

    # n_threads = 2
    iterable = np.arange(10)
    n_threads = 2
    cs = get_chunks(iterable, n_threads)
    assert len(cs) == 2
    assert set(map(len, cs)) == {5}
    assert {x for i in cs for x in i} == set(iterable)

    # n_threads = 10
    iterable = np.arange(10)
    n_threads = 10
    cs = get_chunks(iterable, n_threads)
    assert len(cs) == 10
    assert set(map(len, cs)) == {1}
    assert {x for i in cs for x in i} == set(iterable)


def test_cm_data_is_binary_evenly_distributed():
    # Prepare
    np.random.seed(0)

    # two features with a quadratic relationship
    feature0 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    feature1 = np.random.rand(10)

    # Run
    cm_value, max_parts, parts = ccc(
        feature0, feature1, internal_n_clusters=[2], return_parts=True
    )

    # Validate
    assert cm_value < 0.05

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].shape == (1, 10)

    # the partition should separate true from false values in data
    assert ari(parts[0][0], feature0) == 1.0


def test_cm_data_is_binary_not_evenly_distributed():
    # Prepare
    np.random.seed(0)

    # two features with a quadratic relationship
    feature0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    feature1 = np.random.rand(10)

    # Run
    cm_value, max_parts, parts = ccc(
        feature0, feature1, internal_n_clusters=[2], return_parts=True
    )

    # Validate
    assert cm_value < 0.05

    assert parts is not None
    assert len(parts) == 2
    assert parts[0].shape == (1, 10)

    # the partition should separate true from false values in data
    assert ari(parts[0][0], feature0) == 1.0


def test_cm_numerical_and_categorical_features_perfect_relationship():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects
    numerical_feature0 = np.random.rand(100)
    numerical_feature0_median = np.percentile(numerical_feature0, 50)

    # create a categorical variable perfectly correlated with the numerical one (this is actually an ordinal feature)
    categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.str_)
    categorical_feature1[numerical_feature0 < numerical_feature0_median] = "l"
    categorical_feature1[numerical_feature0 >= numerical_feature0_median] = "u"
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 2
    assert set(_unique_values) == {"l", "u"}

    # Run
    cm_value = ccc(numerical_feature0, categorical_feature1)
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == 1.0

    # flip variables (symmetry)
    assert ccc(categorical_feature1, numerical_feature0) == cm_value


def test_cm_numerical_and_categorical_features_strong_relationship():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects
    numerical_feature0 = np.random.rand(100)
    numerical_feature0_perc = np.percentile(numerical_feature0, 25)

    # create a categorical variable perfectly correlated with the numerical one (this is actually an ordinal feature)
    categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.str_)
    categorical_feature1[numerical_feature0 < numerical_feature0_perc] = "l"
    categorical_feature1[numerical_feature0 >= numerical_feature0_perc] = "u"
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 2
    assert set(_unique_values) == {"l", "u"}

    # Run
    cm_value = ccc(numerical_feature0, categorical_feature1)
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.326, abs=0.001)

    # flip variables (symmetry)
    assert ccc(categorical_feature1, numerical_feature0) == cm_value


def test_cm_numerical_and_categorical_features_no_relationship():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects
    numerical_feature0 = np.random.rand(100)

    # create a categorical variable perfectly correlated with the numerical one (this is actually an ordinal feature)
    categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.str_)
    categorical_feature1[numerical_feature0 < 0.50] = "l"
    categorical_feature1[numerical_feature0 >= 0.50] = "u"
    np.random.shuffle(categorical_feature1)
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 2
    assert set(_unique_values) == {"l", "u"}

    # Run
    cm_value = ccc(numerical_feature0, categorical_feature1)
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.01, abs=0.01)

    # flip variables (symmetry)
    assert ccc(categorical_feature1, numerical_feature0) == cm_value


def test_cm_numerical_and_categorical_features_too_many_categories():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects
    numerical_feature0 = np.random.rand(100)

    # create a categorical variable perfectly correlated with the numerical one (this is actually an ordinal feature)
    categorical_feature1 = np.full(numerical_feature0.shape[0], "cat100", dtype="S6")
    for idx in range(categorical_feature1.shape[0]):
        categorical_feature1[idx] = f"cat{idx:d}"
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 100

    # Run
    cm_value = ccc(numerical_feature0, categorical_feature1)
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == 0.0

    # flip variables (symmetry)
    assert ccc(categorical_feature1, numerical_feature0) == cm_value


def test_cm_numerical_and_categorical_features_a_single_categorical_value():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects
    numerical_feature0 = np.random.rand(100)

    # create a categorical variable perfectly correlated with the numerical one (this is actually an ordinal feature)
    categorical_feature1 = np.full(numerical_feature0.shape[0], "c", dtype="S1")
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 1

    # Run
    cm_value = ccc(numerical_feature0, categorical_feature1)
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == 0.0

    # flip variables (symmetry)
    assert ccc(categorical_feature1, numerical_feature0) == cm_value


def test_cm_numerical_and_categorical_features_with_pandas_dataframe_two_features():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects
    numerical_feature0 = np.random.rand(100)
    numerical_feature0_median = np.percentile(numerical_feature0, 50)

    # create a categorical variable perfectly correlated with the numerical one (this is actually an ordinal feature)
    categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.str_)
    categorical_feature1[numerical_feature0 < numerical_feature0_median] = "l"
    categorical_feature1[numerical_feature0 >= numerical_feature0_median] = "u"
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 2
    assert set(_unique_values) == {"l", "u"}

    data = pd.DataFrame(
        {
            "numerical_feature": numerical_feature0,
            "categorical_feature": categorical_feature1,
        }
    )

    # Run
    cm_value = ccc(data)
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == 1.0

    # flip variables (symmetry)
    assert ccc(data.iloc[:, [1, 0]]) == cm_value


def test_cm_numpy_array_input():
    # Prepare
    np.random.seed(123)

    # here I force
    data = np.random.rand(20, 100)

    # Run
    cm_value = ccc(data)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, np.ndarray)
    assert cm_value.shape == (int(data.shape[0] * (data.shape[0] - 1) / 2),)
    assert np.issubdtype(cm_value.dtype, float)


@pytest.mark.skipif(os.cpu_count() < 2, reason="requires at least 2 cores")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cm_numpy_array_input_with_n_jobs():
    # Prepare
    np.random.seed(123)

    # here I force
    data = np.random.rand(100, 1000)

    # Run
    start_time = time.time()
    res0 = ccc(data, n_jobs=1)
    elapsed_time_single_thread = time.time() - start_time

    start_time = time.time()
    res1 = ccc(data, n_jobs=2)
    elapsed_time_multi_thread = time.time() - start_time

    # Validate
    assert elapsed_time_multi_thread < 0.75 * elapsed_time_single_thread

    assert res0 is not None
    assert isinstance(res0, np.ndarray)
    assert res0.shape == (int(data.shape[0] * (data.shape[0] - 1) / 2),)
    assert np.issubdtype(res0.dtype, float)
    np.testing.assert_array_equal(res0, res1)


@pytest.mark.skipif(os.cpu_count() < 2, reason="requires at least 2 cores")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cm_two_features_input_with_n_jobs():
    # Prepare
    np.random.seed(123)

    # here I force
    x = np.random.rand(100000)
    y = np.random.rand(100000)

    # Run
    start_time = time.time()
    res0 = ccc(x, y, n_jobs=1)
    elapsed_time_single_thread = time.time() - start_time

    start_time = time.time()
    res1 = ccc(x, y, n_jobs=2)
    elapsed_time_multi_thread = time.time() - start_time

    # Validate
    assert elapsed_time_multi_thread < 0.75 * elapsed_time_single_thread

    assert res0 is not None
    assert isinstance(res0, float)
    assert res0 == res1


@pytest.mark.skipif(os.cpu_count() < 2, reason="requires at least 2 cores")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cm_two_features_input_with_n_jobs_using_threads_for_partitioning():
    # Prepare
    np.random.seed(123)

    # here I force
    x = np.random.rand(100000)
    y = np.random.rand(100000)

    # Run
    start_time = time.time()
    res0 = ccc(x, y, n_jobs=1)
    elapsed_time_single_thread = time.time() - start_time

    start_time = time.time()
    res1 = ccc(x, y, n_jobs=2, partitioning_executor="thread")
    elapsed_time_multi_thread = time.time() - start_time

    # Validate
    assert elapsed_time_multi_thread < 0.75 * elapsed_time_single_thread

    assert res0 is not None
    assert isinstance(res0, float)
    assert res0 == res1


@pytest.mark.skipif(os.cpu_count() < 2, reason="requires at least 2 cores")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cm_two_features_input_with_n_jobs_using_process_for_partitioning():
    # Prepare
    np.random.seed(123)

    # here I force
    x = np.random.rand(100000)
    y = np.random.rand(100000)

    # Run
    start_time = time.time()
    res0 = ccc(x, y, n_jobs=1)
    elapsed_time_single_thread = time.time() - start_time

    start_time = time.time()
    res1 = ccc(x, y, n_jobs=2, partitioning_executor="process")
    elapsed_time_multi_thread = time.time() - start_time

    # Validate
    # less stringent than with threads, because the overhead of using processes
    # seems to be larger
    assert elapsed_time_multi_thread < elapsed_time_single_thread

    assert res0 is not None
    assert isinstance(res0, float)
    assert res0 == res1


def test_cm_with_pandas_dataframe_several_features():
    # Prepare
    np.random.seed(123)

    # here I force
    data = pd.DataFrame(np.random.rand(20, 100))

    # Run
    cm_value = ccc(data, internal_n_clusters=3)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, np.ndarray)
    assert cm_value.shape == (int(data.shape[1] * (data.shape[1] - 1) / 2),)
    assert np.issubdtype(cm_value.dtype, float)


def test_cm_with_too_few_objects():
    # Prepare
    np.random.seed(123)

    # here I force
    data = np.random.rand(10, 2)

    # Run
    with pytest.raises(ValueError) as e:
        ccc(data, internal_n_clusters=3)

    assert "too few objects" in str(e.value)



@pytest.mark.parametrize("n_jobs, cpu_count, expected", [
    (None, 4, 4),
    (2, 4, 2),
    (-1, 4, 3),
    (6, 4, 6),
])
def test_get_n_workers_valid(n_jobs, cpu_count, expected):
    with patch('os.cpu_count', return_value=cpu_count):
        assert get_n_workers(n_jobs) == expected


@pytest.mark.parametrize("n_jobs, cpu_count, error_type, error_message", [
    (0, 4, ValueError, "The number of threads/processes to use must be greater than 0. Got 0"),
    (-5, 4, ValueError, "The number of threads/processes to use must be greater than 0. Got -1"),
    (2, None, ValueError, "Could not determine the number of CPU cores. Please specify a positive value of n_jobs"),
    (None, None, ValueError, "Could not determine the number of CPU cores. Please specify a positive value of n_jobs"),
])
def test_get_n_workers_invalid(n_jobs, cpu_count, error_type, error_message):
    with patch('os.cpu_count', return_value=cpu_count):
        with pytest.raises(error_type, match=error_message):
            get_n_workers(n_jobs)
