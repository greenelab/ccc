import pytest
from typing import List

import numpy as np


from ccc.coef.impl_gpu import get_perc_from_k, get_range_n_percs


def test_get_perc_from_k_with_k_less_than_two():
    empty_array = np.empty(0)
    np.testing.assert_array_equal(get_perc_from_k(1), empty_array)
    np.testing.assert_array_equal(get_perc_from_k(0), empty_array)
    np.testing.assert_array_equal(get_perc_from_k(-1), empty_array)


def test_get_perc_from_k():
    assert get_perc_from_k(2) == [0.5]
    assert np.round(get_perc_from_k(3), 3).tolist() == [0.333, 0.667]
    assert get_perc_from_k(4) == [0.25, 0.50, 0.75]


def test_get_range_n_percs_basic():
    ks = [2, 3, 4]
    expected: List[List[float]] = [
        [0.5],
        [0.3333333333333333, 0.6666666666666666],
        [0.25, 0.5, 0.75],
        []
    ]
    result = get_range_n_percs(ks)
    assert np.allclose(result, expected)


def test_get_range_n_percs_empty():
    ks: List[int] = []
    expected: List[List[float]] = []
    result = get_range_n_percs(ks)
    assert result == expected


def test_get_range_n_percs_single():
    ks = [1, 0, -1]
    expected = [[], [], []]
    result = get_range_n_percs(ks)
    assert result == expected


def test_get_range_n_percs_large():
    ks = [10, 5, 2]
    expected = [
        [0.1 * i for i in range(1, 10)],
        [0.2, 0.4, 0.6, 0.8],
        [0.5]
    ]
    result = get_range_n_percs(ks)
    assert result == expected


def test_get_range_n_percs_mixed():
    ks = [4, 3, 0, 1, 5]
    expected = [
        [0.25, 0.5, 0.75],
        [0.3333333333333333, 0.6666666666666666],
        [],
        [],
        [0.2, 0.4, 0.6, 0.8]
    ]
    result = get_range_n_percs(ks)
    assert result == expected
