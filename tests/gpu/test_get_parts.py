import pytest

import numpy as np
import cupy as cp

from ccc.coef.impl_gpu import (
    get_parts,
)

from ccc.coef import get_parts as get_parts_cpu
from ccc.coef import get_perc_from_k as get_perc_from_k_cpu
import functools

def clean_gpu_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    return wrapper

def find_partition(value, quantiles):
    for i in range(len(quantiles)):
        if value <= quantiles[i]:
            return i
    return len(quantiles)  # If value is greater than all quantiles

def verify_partition(feature, index, n_clusters):
    """
    Verify the partition for a specific element in the feature array.
    """
    parts_cpu = get_parts_cpu(feature, (n_clusters,))
    percentages_cpu = get_perc_from_k_cpu(n_clusters)
    quantities = np.quantile(feature, percentages_cpu)
    
    value = feature[index]
    partition = find_partition(value, quantities)
    
    print(f"\nVerifying partition for feature[{index}] = {value}")
    print(f"CPU percentages: {percentages_cpu}")
    print(f"CPU quantities: {quantities}")

    print("\nAll partition ranges:")
    for i in range(n_clusters):
        if i == 0:
            print(f"Partition {i} range: (-inf, {quantities[i]}]")
        elif i == n_clusters - 1:
            print(f"Partition {i} range: ({quantities[i-1]}, inf)")
        else:
            print(f"Partition {i} range: ({quantities[i-1]}, {quantities[i]}]")

    print(f"Data point {value} should fall in partition {partition}")
    print(f"Partition computed by CCC_CPU: {parts_cpu[0][index]}")
    
    assert partition == parts_cpu[0][index], f"Mismatch in partition for feature[{index}]"
    return partition


@clean_gpu_memory
@pytest.mark.parametrize("feature_size", [100, 1000, 10000, 100000])
@pytest.mark.parametrize("cluster_settings", [
    ([2], (2,)),
    ([2, 3], (2, 3)),
    ([2, 3, 4], (2, 3, 4)),
    ([5], (5,)),
    ([6], (6,)),
    ([9], (9,)),
    ([2, 3, 4, 5, 6, 7, 8, 9, 10], (2, 3, 4, 5, 6, 7, 8, 9, 10)),
])
@pytest.mark.parametrize("seed, distribution, params", [
    (0, "rand", {}),  # Uniform distribution
    (42, "randn", {}),  # Normal distribution
    (123, "randint", {"low": 0, "high": 100}),  # Integer distribution
    (456, "exponential", {"scale": 2.0}),  # Exponential distribution
])
def test_get_parts(feature_size, cluster_settings, seed, distribution, params):
    # Given FP arithmetic is not associative and the difference between GPU and CPU FP arithmetic,
    # we need to allow for some tolerance. This is a tentative value that may need to be adjusted.
    # Note that the difference between GPU and CPU results is not expected to be larger than 1.
    n_diff_tolerance = int(feature_size * 0.04)

    np.random.seed(seed)
    
    gpu_clusters, cpu_clusters = cluster_settings

    # Generate random features based on the specified distribution
    if distribution == "rand":
        feature = np.random.rand(feature_size)
    elif distribution == "randn":
        feature = np.random.randn(feature_size)
    elif distribution == "randint":
        feature = np.random.randint(params["low"], params["high"], feature_size)
    elif distribution == "exponential":
        feature = np.random.exponential(params["scale"], feature_size)
    elif distribution == "binomial":
        feature = np.random.binomial(params["n"], params["p"], feature_size)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    # GPU implementation
    parts_gpu = get_parts(feature, np.array(gpu_clusters, dtype=np.uint8))[0].get()
    
    # CPU implementation
    parts_cpu = get_parts_cpu(feature, cpu_clusters)

    print(f"\nTesting with feature_size={feature_size}, clusters={gpu_clusters}, distribution={distribution}")
    print(f"GPU output shape: {parts_gpu.shape}")
    print(f"CPU output shape: {parts_cpu.shape}")
    
    assert parts_gpu is not None, "GPU output is None"
    assert len(parts_gpu) == 1, f"Expected 1 feature, got {len(parts_gpu)}"
    assert len(parts_gpu[0]) == len(gpu_clusters), f"Expected {len(gpu_clusters)} partition(s), got {len(parts_gpu[0])}"
    
    for i, n_clusters in enumerate(gpu_clusters):
        gpu_unique = np.unique(parts_gpu[0][i])
        cpu_unique = np.unique(parts_cpu[i])
        
        print(f"\nPartition {i}:")
        print(f"  GPU unique values (partitions): {gpu_unique}")
        print(f"  CPU unique values (partitions): {cpu_unique}")
        
        assert len(gpu_unique) == n_clusters, f"Expected {n_clusters} cluster indexes, got {len(gpu_unique)}"
        
        if not np.array_equal(parts_gpu[0][i], parts_cpu[i]):
            diff_indices = np.where(parts_gpu[0][i] != parts_cpu[i])[0]
            diff_values = np.abs(parts_gpu[0][i][diff_indices] - parts_cpu[i][diff_indices])
            max_diff = np.max(diff_values)
            
            print(f"\nDifferences found in partition {i}:")
            print(f"  Number of differing elements: {len(diff_indices)}")
            print(f"  Maximum difference: {max_diff}")
            print(f"  First 10 differing indices: {diff_indices[:10]}")
            print(f"  GPU values at these indices: {parts_gpu[0][i][diff_indices[:10]]}")
            print(f"  CPU values at these indices: {parts_cpu[i][diff_indices[:10]]}")
            print(f"  Object values at these indices: {feature[diff_indices[:10]]}")
            
            if len(diff_indices) > n_diff_tolerance or max_diff > 1:
                # Verify partitions for differing elements
                for idx in diff_indices[:10]:
                    expected_partition = verify_partition(feature, idx, n_clusters)
                    assert parts_gpu[0][i][idx] == expected_partition, f"GPU partition mismatch for feature[{idx}]"
                
                assert False, f"GPU and CPU results don't match for {n_clusters} clusters: " \
                              f"diff count = {len(diff_indices)}, max diff = {max_diff}"
            else:
                print(f"  Differences within tolerance (count <= {n_diff_tolerance} and max diff <= 1)")
    
    # Additional checks for multi-cluster settings
    if len(gpu_clusters) > 1:
        for i in range(len(gpu_clusters)):
            for j in range(i + 1, len(gpu_clusters)):
                if np.array_equal(parts_gpu[0][i], parts_cpu[j]):
                    print(f"\nUnexpected equality between partitions {i} and {j}:")
                    print(f"  Partition {i}: {parts_gpu[0][i]}")
                    print(f"  Partition {j}: {parts_cpu[j]}")
                    assert False, f"Partitions {i} and {j} should not be equal"


def test_specific_elements():
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    np.random.seed(0)
    feature = np.random.rand(100)
    assert feature[77] == 0.1201965612131689
    assert feature[78] == 0.29614019752214493
    
    verify_partition(feature, 77, 6)
    verify_partition(feature, 78, 6)


@clean_gpu_memory
def test_potential_buggy_cpu_impl():

    np.random.seed(0)
    feature = np.random.rand(100)
    assert feature[77] == 0.1201965612131689
    assert feature[78] == 0.29614019752214493
    parts_cpu = get_parts_cpu(feature, (6, ))
    percentages_cpu = get_perc_from_k_cpu(6)
    quantities = np.quantile(feature, percentages_cpu)
    print()
    print(f"CPU parts: \n{parts_cpu}")
    print(f"CPU percentages: \n{percentages_cpu}")
    print(f"CPU quantities: \n{quantities}")

    # Find which partitions feature[77] and feature[78] fall into
    value_77 = feature[77]
    value_78 = feature[78]
    partition_77 = find_partition(value_77, quantities)
    partition_78 = find_partition(value_78, quantities)

    print(f"feature[77] = {value_77} falls in partition {partition_77}")
    print(f"feature[78] = {value_78} falls in partition {partition_78}")
    if partition_77 > 0:
        print(f"Partition {partition_77} range: ({quantities[partition_77-1]}, {quantities[partition_77]}]")
    else:
        print(f"Partition {partition_77} range: (-inf, {quantities[partition_77]}]")
    if partition_78 > 0:
        print(f"Partition {partition_78} range: ({quantities[partition_78-1]}, {quantities[partition_78]}]")
    else:
        print(f"Partition {partition_78} range: (-inf, {quantities[partition_78]}]")
    print(f"Partition computed by CCC_CPU for feature[77]: {parts_cpu[0][77]}")
    print(f"Partition computed by CCC_CPU for feature[78]: {parts_cpu[0][78]}")
    assert partition_77 == parts_cpu[0][77]
    assert partition_78 == parts_cpu[0][78]


@clean_gpu_memory
def test_get_parts_with_singletons():

    np.random.seed(0)

    feature0 = np.array([1.3] * 100)

    # run
    parts = get_parts(feature0, np.array([2], dtype=np.uint8))[0].get()
    parts_cpu = get_parts_cpu(feature0, (2,))
    assert parts is not None
    assert len(parts) == 1 # 1 feature
    assert len(parts[0]) == 1 # 1 partition
    # all the elements (2D) should be -2
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([-2]))
    assert np.array_equal(parts[0], parts_cpu)

    parts = get_parts(feature0, np.array([2, 3], dtype=np.uint8))[0].get()
    parts_cpu = get_parts_cpu(feature0, (2, 3))
    assert parts is not None
    assert len(parts) == 1
    assert len(parts[0]) == 2, "feature should have 2 clusters"
    np.testing.assert_array_equal(np.unique(parts[0][0]), np.array([-2]))
    np.testing.assert_array_equal(np.unique(parts[0][1]), np.array([-2]))
    assert np.array_equal(parts[0][0], parts_cpu[0])
    assert np.array_equal(parts[0][1], parts_cpu[1])


@clean_gpu_memory
def test_get_parts_with_categorical_feature():
    np.random.seed(0)

    feature0 = np.array([4] * 10)

    # run
    # only one partition is requested
    parts = get_parts(feature0, np.array([2], dtype=np.uint8), data_is_numerical=False)[0].get()
    parts_cpu = get_parts_cpu(feature0, (2,), data_is_numerical=False)
    assert parts is not None
    assert len(parts) == 1
    assert len(parts[0]) == 1
    np.testing.assert_array_equal(np.unique(parts[0]), np.array([4]))
    assert np.array_equal(parts[0], parts_cpu)

    # more partitions are requested; only the first one has valid information
    parts = get_parts(feature0, np.array([2, 3], dtype=np.uint8), data_is_numerical=False)[0].get()
    parts_cpu = get_parts_cpu(feature0, (2, 3), data_is_numerical=False)
    assert parts is not None
    assert len(parts) == 1
    assert len(parts[0]) == 2
    print("parts:")
    print(parts)
    print("parts_cpu:")
    print(parts_cpu)
    np.testing.assert_array_equal(np.unique(parts[0][0]), np.array([4]))
    np.testing.assert_array_equal(np.unique(parts[0][1]), np.array([-1]))
    assert (parts == parts_cpu).all()
    assert np.array_equal(parts[0][0], parts_cpu[0])
    assert np.array_equal(parts[0][1], parts_cpu[1])


@clean_gpu_memory
def test_get_parts_2d_simple():
    np.random.seed(0)
    array = np.random.rand(5, 1000)
    print(f"array : \n{array}")
    parts = get_parts(array, np.array([3], dtype=np.uint8))[0].get()
    parts_cpu_row0 = get_parts_cpu(array[0], (3, ))
    parts_cpu_row1 = get_parts_cpu(array[1], (3, ))
    assert parts is not None
    assert (parts[0] == parts_cpu_row0).all()
    assert (parts[1] == parts_cpu_row1).all()
    print("parts:")
    print(parts)
    print("parts_cpu_row0:")
    print(parts_cpu_row0)
    print("parts_cpu_row1:")
    print(parts_cpu_row1)
