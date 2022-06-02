import numpy as np
from scipy import stats

from ccc.scipy.stats import rank


def test_rank_no_duplicates():
    data = np.array([0, 10, 1, 5, 7, 8, -5, -2])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group():
    data = np.array([0, 10, 1, 5, 7, 8, 1, -2])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_with_more_elements():
    data = np.array([0, 10, 1, 1, 7, 8, 1, -2])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_beginning():
    data = np.array([0, 0, 1, -10, 7, 8, 9.4, -2])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_beginning_with_more_elements():
    data = np.array([0.13, 0.13, 0.13, 1, -10, 7, 8, 9.4, -2])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_beginning_are_smallest():
    data = np.array([0, 10, 1.5, -99.5, -99.5, -99.5, 5, 7, 8, -5, -2])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_end():
    data = np.array([0, 1, -10, 7, 8, 9.4, -2.5, -2.5])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_end_with_more_elements():
    data = np.array([0, 1, -10, 7, 8, 9.4, -12.5, -12.5, -12.5])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_one_duplicate_group_at_end_is_the_largest():
    data = np.array([0, 1, -10, 7, 8, 9.4, 120.5, 120.5, 120.5])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)


def test_rank_all_are_duplicates():
    data = np.array([1.5, 1.5, 1.5, 1.5])

    expected_ranks = stats.rankdata(data, "average")
    observed_ranks = rank(data)

    np.testing.assert_array_equal(observed_ranks, expected_ranks)
