import pytest
import numpy as np
from ccc.sklearn.metrics_device import find_unique, compute_contingency_matrix, get_pair_confusion_matrix, sum_2d_array, sum_squares_2d_array, adjusted_rand_index, compute_ari


# Define the maximum unique values for testing
MAX_UNIQUE = 10
MAX_CLUSTERS = 5


# Helper function to run device functions in tests
def run_device_function(func, *args):
    """Helper to run a CUDA device function."""
    out = func(*args)
    return out


# Test for find_unique
def test_find_unique():
    arr = np.array([1, 2, 2, 3, 4, 4, 4, 5], dtype=np.int32)
    expected_unique = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    expected_counts = np.array([1, 2, 1, 3, 1], dtype=np.int32)

    unique, counts, num_unique = run_device_function(find_unique, arr, MAX_UNIQUE)

    assert num_unique == len(expected_unique)
    assert np.all(unique == expected_unique)
    assert np.all(counts == expected_counts)


# Test for compute_contingency_matrix
def test_compute_contingency_matrix():
    part0 = np.array([0, 1, 1, 2], dtype=np.int32)
    part1 = np.array([1, 1, 0, 2], dtype=np.int32)

    cont_mat = np.zeros((MAX_CLUSTERS, MAX_CLUSTERS), dtype=np.int32)
    num_clusters0, num_clusters1 = run_device_function(compute_contingency_matrix, part0, part1, cont_mat, MAX_CLUSTERS)

    expected_cont_mat = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ], dtype=np.int32)

    assert np.all(cont_mat[:num_clusters0, :num_clusters1] == expected_cont_mat)


# Test for sum_2d_array
def test_sum_2d_array():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    total = run_device_function(sum_2d_array, arr, 2, 3)

    assert total == 21  # Sum of all elements in arr


# Test for sum_squares_2d_array
def test_sum_squares_2d_array():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    total_squares = run_device_function(sum_squares_2d_array, arr, 2, 3)

    assert total_squares == 91  # Sum of squares of all elements in arr


# Test for get_pair_confusion_matrix
def test_get_pair_confusion_matrix():
    part0 = np.array([0, 1, 1, 2], dtype=np.int32)
    part1 = np.array([1, 1, 0, 2], dtype=np.int32)

    C = run_device_function(get_pair_confusion_matrix, part0, part1, MAX_CLUSTERS)

    assert C[0, 0] == 0  # Example check for specific value in the confusion matrix


# Test for adjusted_rand_index
def test_adjusted_rand_index():
    part0 = np.array([0, 1, 1, 2], dtype=np.int32)
    part1 = np.array([1, 1, 0, 2], dtype=np.int32)

    # Expected ARI between these partitions is some value we calculate manually or use sklearn for comparison
    out = np.zeros((1, 1, 1), dtype=np.float32)

    run_device_function(adjusted_rand_index, part0, part1, out, 0, 0, 0, MAX_CLUSTERS)

    assert out[0, 0, 0] == pytest.approx(0.4444, rel=1e-4)  # Example value based on expected ARI


# Test for compute_ari kernel
def test_compute_ari_kernel():
    partitions = np.array([[[0, 1, 1], [1, 0, 0]]], dtype=np.int32)
    out = np.zeros((1, 2, 2), dtype=np.float32)

    compute_ari[1, 1](partitions, out, MAX_CLUSTERS)

    # Example check for ARI result
    assert out[0, 0, 1] == pytest.approx(0.3333, rel=1e-4)

