from random import shuffle

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import adjusted_rand_score as ari

from clustermatch.coef import (
    cm,
    _get_range_n_clusters,
    run_quantile_clustering,
    _get_perc_from_k,
    _get_parts,
    rank,
    cdist_parts,
    get_coords_from_index,
    unravel_index_2d,
)


def test_rank_no_duplicates():
    data = np.array([0, 10, 1, 5, 7, 8, -5, -2])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group():
    data = np.array([0, 10, 1, 5, 7, 8, 1, -2])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_with_more_elements():
    data = np.array([0, 10, 1, 1, 7, 8, 1, -2])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_beginning():
    data = np.array([0, 0, 1, -10, 7, 8, 9.4, -2])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_beginning_with_more_elements():
    data = np.array([0.13, 0.13, 0.13, 1, -10, 7, 8, 9.4, -2])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_beginning_are_smallest():
    data = np.array([0, 10, 1.5, -99.5, -99.5, -99.5, 5, 7, 8, -5, -2])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_end():
    data = np.array([0, 1, -10, 7, 8, 9.4, -2.5, -2.5])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_end_with_more_elements():
    data = np.array([0, 1, -10, 7, 8, 9.4, -12.5, -12.5, -12.5])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_end_is_the_largest():
    data = np.array([0, 1, -10, 7, 8, 9.4, 120.5, 120.5, 120.5])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_all_are_duplicates():
    data = np.array([1.5, 1.5, 1.5, 1.5])

    expected_ranks = stats.rankdata(data, "dense")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


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
    np.testing.assert_array_equal(
        range_n_clusters, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    )

    # 25 features
    range_n_clusters = _get_range_n_clusters(25)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4, 5]))


def test_get_range_n_clusters_with_internal_n_clusters_is_list():
    # 100 features
    range_n_clusters = _get_range_n_clusters(
        100,
        internal_n_clusters=[
            2,
        ],
    )
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))

    # 25 features
    range_n_clusters = _get_range_n_clusters(
        25,
        internal_n_clusters=[
            2,
        ],
    )
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=[2, 3, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4]))


def test_get_range_n_clusters_with_internal_n_clusters_none():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=None)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(
        range_n_clusters, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    )

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=None)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4, 5]))


def test_get_range_n_clusters_with_internal_n_clusters_has_single_int():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[2])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))

    # 25 features
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=[3])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([3]))

    # 5 features
    range_n_clusters = _get_range_n_clusters(5, internal_n_clusters=[4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([4]))

    # 25 features but invalid number of clusters
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=[1])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))

    # 25 features but invalid number of clusters
    range_n_clusters = _get_range_n_clusters(25, internal_n_clusters=[25])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))


def test_get_range_n_clusters_with_internal_n_clusters_are_less_than_two():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 3, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4]))

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 1, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 4]))

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 3, 1])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3]))

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 0, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 4]))

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[1, 2, 1, -4, 6])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 6]))


def test_get_range_n_clusters_with_internal_n_clusters_are_repeated():
    # 100 features
    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[2, 3, 2, 4])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2, 3, 4]))

    range_n_clusters = _get_range_n_clusters(100, internal_n_clusters=[2, 2, 2])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))


def test_get_range_n_clusters_with_very_few_features():
    # 3 features
    range_n_clusters = _get_range_n_clusters(3)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([2]))

    # 2 features
    range_n_clusters = _get_range_n_clusters(2)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))

    # 1 features
    range_n_clusters = _get_range_n_clusters(1)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))

    # 0 features
    range_n_clusters = _get_range_n_clusters(0)
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))


def test_get_range_n_clusters_with_larger_k_than_features():
    # 10 features
    range_n_clusters = _get_range_n_clusters(10, internal_n_clusters=[10])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))

    # 10 features
    range_n_clusters = _get_range_n_clusters(10, internal_n_clusters=[11])
    assert range_n_clusters is not None
    np.testing.assert_array_equal(range_n_clusters, np.array([]))


def test_get_range_n_clusters_with_default_max_k():
    range_n_clusters = _get_range_n_clusters(200)
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
    cm_value = cm(feature0, feature1)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, float)


def test_cm_ari_is_negative():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    feature0 = np.array([1, 2, 3, 4, 5])
    feature1 = np.array([2, 4, 1, 3, 5])

    # Run
    cm_value = cm(feature0, feature1)

    # Validate
    # ari for this example is -0.25, but cm should return 0.0
    assert cm_value == 0.0


def test_cm_random_data():
    # Prepare
    np.random.seed(123)

    for i in range(10):
        # two features on 100 objects (random data)
        feature0 = minmax_scale(
            np.random.rand(100), (-1.0, 1.0)
        )  # with negative values
        feature1 = np.random.rand(100)  # all positive values between 0 and 1

        # Run
        cm_value = cm(feature0, feature1)

        # Validate
        assert 0.0 <= cm_value < 0.05


def test_cm_linear():
    # Prepare
    np.random.seed(0)

    # two features on 100 objects with a linear relationship
    feature0 = np.random.rand(100)
    feature1 = feature0 * 5.0

    # Run
    cm_value = cm(feature0, feature1)

    # Validate
    assert cm_value == 1.0


def test_cm_quadratic():
    # Prepare
    np.random.seed(1)

    # two features on 100 objects with a linear relationship
    feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0)

    # Run
    cm_value = cm(feature0, feature1)

    # Validate
    assert cm_value > 0.40


def test_cm_quadratic2():
    # Prepare
    np.random.seed(1)

    # two features on 100 objects with a linear relationship
    feature0 = minmax_scale(np.random.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0) + (0.10 * np.random.rand(feature0.shape[0]))

    # Run
    cm_value = cm(feature0, feature1)

    # Validate
    assert cm_value > 0.40


def test_cm_one_feature_with_all_same_values():
    # if there is no variation in at least one of the two variables to be
    #  compared, clustermatch returns nan

    # Prepare
    np.random.seed(0)

    # two features on 100 objects; all values in feature1 are the same
    feature0 = np.random.rand(100)
    feature1 = np.array([5] * feature0.shape[0])

    # Run
    cm_value = cm(feature0, feature1)

    # Validate
    assert np.isnan(cm_value), cm_value


def test_cm_all_features_with_all_same_values():
    # if there is no variation in both variables to be compared, clustermatch
    #  returns nan

    # Prepare
    np.random.seed(0)

    # two features with constant values
    feature0 = np.array([0] * 100)
    feature1 = np.array([5] * feature0.shape[0])

    # Run
    cm_value = cm(feature0, feature1)

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
    cm_value = cm(input_data)

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
    cm_value = cm(feature0, feature1)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, float)


def test_cm_x_is_pandas_dataframe():
    # Prepare
    np.random.seed(123)

    # two features on 100 objects (random data)
    data_matrix = pd.DataFrame(np.random.rand(10, 100))

    # Run
    cm_value = cm(data_matrix)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, np.ndarray)
    assert cm_value.shape == (45,)
    assert np.issubdtype(cm_value.dtype, np.float)


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


def test_get_parts_with_singletons():
    np.random.seed(0)

    feature0 = np.array([1.3] * 10)

    # run
    parts = _get_parts(feature0, (2,))
    assert parts is not None
    assert len(parts) == 1
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([-1]))

    parts = _get_parts(feature0, (2, 3))
    assert parts is not None
    assert len(parts) == 2
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([-1]))
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

    observed_cdist = cdist_parts(parts0, parts1)

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

    observed_cdist = cdist_parts(parts0, parts1)

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

    observed_cdist = cdist_parts(parts0, parts1)

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

    observed_cdist = cdist_parts(parts0, parts1)

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
    # compare with results obtained from the original clustermatch
    # implementation (https://github.com/sinc-lab/clustermatch) plus some
    # patches (see tests/data/README.md about clustermatch data).
    from pathlib import Path
    import pandas as pd

    # from pandas.testing import assert_frame_equal

    input_data_dir = Path(__file__).parent / "data"

    # load data
    data = pd.read_pickle(input_data_dir / "clustermatch-random_data-data.pkl")
    data = data.to_numpy()

    # run new clustermatch implementation.
    # Here, I fixed the internal number of clusters, since that slightly changed
    # in the new implementation compared with the original one.
    corr_mat = cm(data, internal_n_clusters=list(range(2, 10 + 1)))

    expected_corr_matrix = pd.read_pickle(
        input_data_dir / "clustermatch-random_data-coef.pkl"
    )
    expected_corr_matrix = expected_corr_matrix.to_numpy()
    expected_corr_matrix = expected_corr_matrix[
        np.triu_indices(expected_corr_matrix.shape[0], 1)
    ]

    np.testing.assert_almost_equal(
        expected_corr_matrix,
        corr_mat,
    )


def test_unravel_index_2d_square_simple():
    shape = (2, 2)
    assert unravel_index_2d(0, shape) == (0, 0)
    assert unravel_index_2d(1, shape) == (0, 1)
    assert unravel_index_2d(2, shape) == (1, 0)
    assert unravel_index_2d(3, shape) == (1, 1)


def test_unravel_index_2d_rect_simple():
    shape = (2, 3)
    assert unravel_index_2d(0, shape) == (0, 0)
    assert unravel_index_2d(1, shape) == (0, 1)
    assert unravel_index_2d(2, shape) == (0, 2)
    assert unravel_index_2d(3, shape) == (1, 0)
    assert unravel_index_2d(4, shape) == (1, 1)
    assert unravel_index_2d(5, shape) == (1, 2)

    shape = (1, 4)
    assert unravel_index_2d(0, shape) == (0, 0)
    assert unravel_index_2d(1, shape) == (0, 1)
    assert unravel_index_2d(2, shape) == (0, 2)
    assert unravel_index_2d(3, shape) == (0, 3)

    shape = (4, 1)
    assert unravel_index_2d(0, shape) == (0, 0)
    assert unravel_index_2d(1, shape) == (1, 0)
    assert unravel_index_2d(2, shape) == (2, 0)
    assert unravel_index_2d(3, shape) == (3, 0)


def test_unravel_index_2d_square0():
    x = np.array([[0, 7], [-5, 6.999]])
    x_max_idx = np.argmax(x, axis=None)
    assert x_max_idx == 1

    expected_idx = np.unravel_index(x_max_idx, x.shape)
    observed_idx = unravel_index_2d(x_max_idx, x.shape)

    assert expected_idx == observed_idx == (0, 1)


def test_unravel_index_2d_square1():
    x = np.array([[0, 7], [-5, 7.01]])
    x_max_idx = np.argmax(x, axis=None)
    assert x_max_idx == 3

    expected_idx = np.unravel_index(x_max_idx, x.shape)
    observed_idx = unravel_index_2d(x_max_idx, x.shape)

    assert expected_idx == observed_idx == (1, 1)


def test_unravel_index_2d_square_all_equal():
    x = np.array([[7.0, 7.0], [7.0, 7.0]])
    x_max_idx = np.argmax(x, axis=None)
    assert x_max_idx == 0

    expected_idx = np.unravel_index(x_max_idx, x.shape)
    observed_idx = unravel_index_2d(x_max_idx, x.shape)

    assert expected_idx == observed_idx == (0, 0)


def test_unravel_index_2d_rect():
    x = np.array([[0, 7, -5.6], [8.1, 6.999, 0]])
    x_max_idx = np.argmax(x, axis=None)
    assert x_max_idx == 3

    expected_idx = np.unravel_index(x_max_idx, x.shape)
    observed_idx = unravel_index_2d(x_max_idx, x.shape)

    assert expected_idx == observed_idx == (1, 0)


def test_unravel_index_index_out_of_bounds():
    with pytest.raises(ValueError):
        unravel_index_2d(6, (2, 3))


def test_unravel_index_non_2d():
    with pytest.raises(ValueError):
        unravel_index_2d(0, (2, 3, 4))


def test_cm_return_parts_quadratic():
    # Prepare
    np.random.seed(0)

    # two features with a quadratic relationship
    feature0 = np.array([-4, -3, -2, -1, 0, 0, 1, 2, 3, 4])
    feature1 = np.array([10, 9, 8, 7, 6, 6, 7, 8, 9, 10])

    # Run
    cm_value, max_parts, parts = cm(
        feature0, feature1, internal_n_clusters=[2, 3], return_parts=True
    )

    # Validate
    assert cm_value.round(2) == 0.59

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
    cm_value, max_parts, parts = cm(feature0, feature1, return_parts=True)

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
