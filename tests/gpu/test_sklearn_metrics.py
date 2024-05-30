import numpy as np
from sklearn.metrics import adjusted_rand_score as sklearn_ari

from ccc.sklearn.metrics_gpu2 import (
    adjusted_rand_index,
    get_contingency_matrix,
    get_pair_confusion_matrix,
)


def test_get_contingency_matrix_k0_equal_k1():
    part0 = np.array([0, 0, 1, 1, 2, 2])
    part1 = np.array([0, 1, 0, 2, 1, 2])

    expected_mat = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])

    observed_mat = get_contingency_matrix(part0, part1)

    np.testing.assert_array_equal(observed_mat, expected_mat)


def test_get_contingency_matrix_k0_greater_k1():
    part0 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    part1 = np.array([0, 1, 0, 2, 1, 2, 2, 2, 2])

    expected_mat = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 3]])

    observed_mat = get_contingency_matrix(part0, part1)

    np.testing.assert_array_equal(observed_mat, expected_mat)


def test_get_contingency_matrix_k0_lesser_k1():
    part0 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 2, 2, 2, 1])
    part1 = np.array([0, 1, 0, 2, 1, 2, 3, 3, 3, 4, 4, 5, 5])

    expected_mat = np.array(
        [[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 2, 1], [0, 0, 0, 3, 0, 0]]
    )

    observed_mat = get_contingency_matrix(part0, part1)

    np.testing.assert_array_equal(observed_mat, expected_mat)


def test_get_pair_confusion_matrix_k0_equal_k1():
    part0 = np.array([0, 0, 1, 1, 2, 2])
    part1 = np.array([0, 1, 0, 2, 1, 2])

    expected_mat = np.array([[18, 6], [6, 0]])

    observed_mat = get_pair_confusion_matrix(part0, part1)

    np.testing.assert_array_equal(observed_mat, expected_mat)


def test_get_pair_confusion_matrix_k0_greater_k1():
    part0 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    part1 = np.array([0, 1, 0, 2, 1, 2, 2, 2, 2])

    expected_mat = np.array([[42, 18], [6, 6]])

    observed_mat = get_pair_confusion_matrix(part0, part1)

    np.testing.assert_array_equal(observed_mat, expected_mat)


def test_adjusted_rand_index_manual_random_partitions_same_k():
    part0 = np.array([0, 0, 1, 1, 2, 2])
    part1 = np.array([0, 1, 0, 2, 1, 2])

    expected_ari = -0.25

    observed_ari = adjusted_rand_index(part0, part1)
    observed_ari_symm = adjusted_rand_index(part1, part0)

    assert observed_ari == observed_ari_symm
    assert expected_ari == observed_ari


def test_adjusted_rand_index_manual_perfect_match():
    part0 = np.array([0, 0, 1, 1, 2, 2])
    part1 = np.array([2, 2, 3, 3, 4, 4])

    expected_ari = 1.0

    observed_ari = adjusted_rand_index(part0, part1)
    observed_ari_symm = adjusted_rand_index(part1, part0)

    assert observed_ari == observed_ari_symm
    assert expected_ari == observed_ari


def test_adjusted_rand_index_random_partitions_same_k():
    maxk0 = 2
    maxk1 = maxk0
    n = 100

    part0 = np.random.randint(0, maxk0 + 1, n)
    part1 = np.random.randint(0, maxk1 + 1, n)

    # warning: the sklearn's ari implementation can overflow in older versions
    # when n is large
    expected_ari = sklearn_ari(part0, part1)

    observed_ari = adjusted_rand_index(part0, part1)
    observed_ari_symm = adjusted_rand_index(part1, part0)

    assert observed_ari == observed_ari_symm
    assert expected_ari == observed_ari


def test_adjusted_rand_index_random_partitions_k0_greater_k1():
    maxk0 = 5
    maxk1 = 3
    n = 100

    part0 = np.random.randint(0, maxk0 + 1, n)
    part1 = np.random.randint(0, maxk1 + 1, n)

    # warning: the sklearn's ari implementation can overflow in older versions
    # when n is large
    expected_ari = sklearn_ari(part0, part1)

    observed_ari = adjusted_rand_index(part0, part1)
    observed_ari_symm = adjusted_rand_index(part1, part0)

    assert observed_ari == observed_ari_symm
    assert expected_ari == observed_ari
