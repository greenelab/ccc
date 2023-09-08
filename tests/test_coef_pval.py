import os
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import minmax_scale

from ccc.coef import ccc

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_cm_basic_pvalue_n_permutations_not_given():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects (random data)
    feature0 = rs.rand(100)
    feature1 = rs.rand(100)

    # Run
    cm_value = ccc(feature0, feature1, pvalue_n_perms=None)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.01, abs=0.01)


def test_cm_basic_pvalue_n_permutations_is_zero():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects (random data)
    feature0 = rs.rand(100)
    feature1 = rs.rand(100)

    # Run
    cm_value = ccc(feature0, feature1, pvalue_n_perms=0)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.01, abs=0.01)


def test_cm_basic_pvalue_n_permutations_is_1():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects (random data)
    feature0 = rs.rand(100)
    feature1 = rs.rand(100)

    # Run
    res = ccc(feature0, feature1, pvalue_n_perms=1)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.01, abs=0.01)

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert 0.0 < pvalue <= 1.0
    assert pvalue in (0.5, 1.0)


def test_cm_basic_pvalue_n_permutations_is_10():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects (random data)
    feature0 = rs.rand(100)
    feature1 = rs.rand(100)

    # Run
    res = ccc(feature0, feature1, pvalue_n_perms=10)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.01, abs=0.01)

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert 0.0 < pvalue <= 1.0


def test_cm_linear_pvalue_n_permutations_10():
    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects with a linear relationship
    feature0 = rs.rand(100)
    feature1 = feature0 * 5.0

    # Run
    res = ccc(feature0, feature1, pvalue_n_perms=10)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == 1.0

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert pvalue == (0 + 1) / (10 + 1)


def test_cm_linear_pvalue_n_permutations_100():
    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects with a linear relationship
    feature0 = rs.rand(100)
    feature1 = feature0 * 5.0

    # Run
    res = ccc(feature0, feature1, pvalue_n_perms=100)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == 1.0

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert pvalue == (0 + 1) / (100 + 1)


def test_cm_quadratic_pvalue():
    # Prepare
    rs = np.random.RandomState(1)

    # two features on 100 objects with a quadratic relationship
    feature0 = minmax_scale(rs.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0)

    # Run
    res = ccc(feature0, feature1, pvalue_n_perms=100)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.49, abs=0.01)

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert pvalue == (0 + 1) / (100 + 1)


def test_cm_quadratic_noisy_pvalue_with_random_state():
    # Prepare
    rs = np.random.RandomState(1)

    # two features on 100 objects with a quadratic relationship
    feature0 = minmax_scale(rs.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0) + (2.0 * rs.rand(feature0.shape[0]))

    # Run
    res = ccc(feature0, feature1, pvalue_n_perms=100, random_state=2)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.05, abs=0.01)

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert pvalue == pytest.approx(0.099, abs=0.01)


def test_cm_one_feature_with_all_same_values_pvalue():
    # if there is no variation in at least one of the two variables to be
    #  compared, ccc returns nan

    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects; all values in feature1 are the same
    feature0 = rs.rand(100)
    feature1 = np.array([5] * feature0.shape[0])

    # Run
    res = ccc(feature0, feature1, pvalue_n_perms=100)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert np.isnan(cm_value), cm_value

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert np.isnan(pvalue), pvalue


def test_cm_single_argument_is_matrix():
    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects with a linear relationship
    feature0 = rs.rand(100)
    feature1 = feature0 * 5.0
    feature2 = rs.rand(feature0.shape[0])

    input_data = np.array([feature0, feature1, feature2])

    # Run
    res = ccc(input_data, pvalue_n_perms=100, random_state=1)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert hasattr(cm_value, "shape")
    assert cm_value.shape == (3,)
    assert cm_value[0] == 1.0
    assert cm_value[1] < 0.03
    assert cm_value[2] < 0.03

    assert pvalue is not None
    assert hasattr(pvalue, "shape")
    assert pvalue.shape == (3,)
    assert pvalue[0] == (0 + 1) / (100 + 1)
    assert pvalue[1] == pytest.approx(0.792, abs=0.01)
    assert pvalue[2] == pytest.approx(0.752, abs=0.01)


@pytest.mark.skipif(os.cpu_count() < 2, reason="requires at least 2 cores")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cm_large_n_objects_pvalue_permutations_is_parallelized():
    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects with a linear relationship
    feature0 = rs.rand(10000)
    feature1 = rs.rand(10000)

    # Run
    start_time = time.time()
    res = ccc(feature0, feature1, pvalue_n_perms=50, n_jobs=1)
    elapsed_time_single_thread = time.time() - start_time

    start_time = time.time()
    res = ccc(feature0, feature1, pvalue_n_perms=50, n_jobs=2)
    elapsed_time_multi_thread = time.time() - start_time

    # Validate
    assert elapsed_time_multi_thread < 0.75 * elapsed_time_single_thread


@pytest.mark.skipif(os.cpu_count() < 2, reason="requires at least 2 cores")
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_cm_medium_n_objects_with_many_pvalue_permutations_is_parallelized():
    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects with a linear relationship
    feature0 = rs.rand(1000)
    feature1 = rs.rand(1000)

    # Run
    start_time = time.time()
    res = ccc(feature0, feature1, pvalue_n_perms=1000, pvalue_n_jobs=1)
    elapsed_time_single_thread = time.time() - start_time

    start_time = time.time()
    res = ccc(feature0, feature1, pvalue_n_perms=1000, pvalue_n_jobs=2)
    elapsed_time_multi_thread = time.time() - start_time

    # Validate
    assert elapsed_time_multi_thread < 0.75 * elapsed_time_single_thread


def test_cm_return_parts_quadratic_pvalue():
    # Prepare
    # rs = np.random.RandomState(0)

    # two features with a quadratic relationship
    feature0 = np.array([-4, -3, -2, -1, 0, 0, 1, 2, 3, 4])
    feature1 = np.array([10, 9, 8, 7, 6, 6, 7, 8, 9, 10])

    # Run
    res, max_parts, parts = ccc(
        feature0,
        feature1,
        internal_n_clusters=[2, 3],
        return_parts=True,
        pvalue_n_perms=10,
    )

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res

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

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert 0.0 < pvalue <= 1.0


def test_cm_numerical_and_categorical_features_perfect_relationship_pvalue():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects
    numerical_feature0 = rs.rand(100)
    numerical_feature0_median = np.percentile(numerical_feature0, 50)

    # create a categorical variable perfectly correlated with the numerical one (this is actually an ordinal feature)
    categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.unicode_)
    categorical_feature1[numerical_feature0 < numerical_feature0_median] = "l"
    categorical_feature1[numerical_feature0 >= numerical_feature0_median] = "u"
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 2
    assert set(_unique_values) == {"l", "u"}

    # Run
    res = ccc(
        numerical_feature0,
        categorical_feature1,
        pvalue_n_perms=100,
    )

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res

    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == 1.0

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert pvalue == (0 + 1) / (100 + 1)

    # Run with flipped variables (symmetry)
    assert ccc(categorical_feature1, numerical_feature0, pvalue_n_perms=100) == res


def test_cm_numerical_and_categorical_features_weakly_relationship_pvalue():
    # if a numerical and categorical vector are flipped and a pvalue is calculated,
    # they do not match the pvalue calculated with the original vector order, because
    # CCC used to flip the second variable; this test makes sure that a more robust
    # strategy is used: the variable that generates more partitions is flipped always

    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects
    numerical_feature0 = rs.rand(100)
    numerical_feature0_perc = np.percentile(numerical_feature0, 2)

    # create a categorical variable strongly correlated with the numerical one
    categorical_feature1 = np.full(numerical_feature0.shape[0], "", dtype=np.unicode_)
    categorical_feature1[numerical_feature0 < numerical_feature0_perc] = "l"
    categorical_feature1[numerical_feature0 >= numerical_feature0_perc] = "u"
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 2
    assert set(_unique_values) == {"l", "u"}

    # Run
    res = ccc(
        categorical_feature1,
        numerical_feature0,
        pvalue_n_perms=100,
        random_state=1,
    )

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res

    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == pytest.approx(0.001, abs=0.001)

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert pvalue == pytest.approx(0.099, abs=0.01)

    # Run with flipped variables (symmetry)
    assert (
        ccc(
            numerical_feature0,
            categorical_feature1,
            pvalue_n_perms=100,
            random_state=1,
        )
        == res
    )


def test_cm_numerical_and_categorical_features_a_single_categorical_value():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects
    numerical_feature0 = rs.rand(100)

    # create a categorical variable with a single value
    categorical_feature1 = np.full(numerical_feature0.shape[0], "c", dtype="S1")
    _unique_values = np.unique(categorical_feature1)
    # some internal checks
    assert _unique_values.shape[0] == 1

    # Run
    res = ccc(
        numerical_feature0,
        categorical_feature1,
        pvalue_n_perms=100,
        random_state=1,
    )

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res

    assert cm_value is not None
    assert isinstance(cm_value, float)
    assert cm_value == 0.0

    assert pvalue is not None
    assert isinstance(pvalue, float)
    assert pvalue == pytest.approx(1.0, abs=0.01)

    # Run with flipped variables (symmetry)
    assert (
        ccc(
            categorical_feature1,
            numerical_feature0,
            pvalue_n_perms=100,
            random_state=1,
        )
        == res
    )


def test_cm_with_pandas_dataframe_several_features():
    # Prepare
    rs = np.random.RandomState(123)

    # here I force
    data = pd.DataFrame(rs.rand(20, 50))

    # Run
    res = ccc(data, internal_n_clusters=3, pvalue_n_perms=10, random_state=1)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res

    assert cm_value is not None
    assert isinstance(cm_value, np.ndarray)
    assert cm_value.shape == (int(50 * (50 - 1) / 2),)
    assert np.issubdtype(cm_value.dtype, float)

    assert pvalue is not None
    assert isinstance(pvalue, np.ndarray)
    assert pvalue.shape == cm_value.shape
    assert np.issubdtype(pvalue.dtype, float)
