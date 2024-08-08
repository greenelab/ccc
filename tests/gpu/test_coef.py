import pytest
from typing import List

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy.typing import NDArray

from ccc.coef.impl_gpu import (
    get_perc_from_k,
    get_range_n_percs,
    convert_n_clusters,
)


def test_get_perc_from_k_with_k_less_than_two():
    empty_array = np.empty(0)
    assert_array_equal(get_perc_from_k(1), empty_array)
    assert_array_equal(get_perc_from_k(0), empty_array)
    assert_array_equal(get_perc_from_k(-1), empty_array)


@pytest.mark.parametrize("k, expected", [
    (2, [0.5]),
    (3, [0.333, 0.667]),
    (4, [0.25, 0.50, 0.75])
])
def test_get_perc_from_k(k, expected):
    assert_allclose(np.ndarray.round(get_perc_from_k(k), 3), expected)


@pytest.mark.parametrize(
    "ks, expected",
    [
        (
                np.array([], dtype=np.int8),
                np.empty((0, 0), dtype=np.float32)
        ),
        (
            np.array([2, 3, 4], dtype=np.int8),
            np.array([
                [0.5, np.nan, np.nan],
                [0.33333334, 0.6666667, np.nan],
                [0.25, 0.5, 0.75]
            ], dtype=np.float32)
        ),
        (
            np.array([2], dtype=np.int8),
            np.array([[0.5]], dtype=np.float32)
        ),
        (
            np.array([10], dtype=np.int8),
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], dtype=np.float32)
        ),
        (
            np.array([2, 4, 6, 8], dtype=np.int8),
            np.array([
                [0.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.25, 0.5, 0.75, np.nan, np.nan, np.nan, np.nan],
                [0.16666667, 0.33333334, 0.5, 0.6666667, 0.8333333, np.nan, np.nan],
                [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
            ], dtype=np.float32)
        ),
        (
            np.array([2, 3, 4], dtype=np.int8),
            np.array([
                [0.5, np.nan, np.nan],
                [0.33333334, 0.6666667, np.nan],
                [0.25, 0.5, 0.75],
            ], dtype=np.float32)
        ),
    ]
)
def test_get_range_n_percs(ks, expected):
    result = get_range_n_percs(ks)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (None, []),
        (2, [2]),
        (5, [2, 3, 4, 5]),
        ([1, 3, 5], [1, 3, 5]),
        ([], []),
        ((7, 8, 9), [7, 8, 9]),
    ]
)
def test_convert_n_clusters(input_value, expected_output):
    assert convert_n_clusters(input_value) == expected_output