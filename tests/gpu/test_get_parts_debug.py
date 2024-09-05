"""
Code to reproduce the edge cased that may be missed by the CPU version of get_parts
"""
import pytest
from typing import List

import numpy as np
from numba import cuda
from numpy.testing import assert_array_equal, assert_allclose
from numpy.typing import NDArray

from ccc.coef.impl_gpu import (
    get_perc_from_k,
    get_range_n_percentages,
    convert_n_clusters,
    get_range_n_clusters,
    get_parts,
)

from ccc.coef import get_parts as get_parts_cpu
from ccc.coef import get_perc_from_k as get_perc_from_k_cpu


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
            print(f"Partition {i} range: ({quantities[i - 1]}, inf)")
        else:
            print(f"Partition {i} range: ({quantities[i - 1]}, {quantities[i]}]")

    print(f"Data point {value} should fall in partition {partition}")
    print(f"Partition computed by CCC_CPU: {parts_cpu[0][index]}")

    assert partition == parts_cpu[0][index], f"Mismatch in partition for feature[{index}]"
    return partition


@pytest.mark.parametrize("feature_size", [100]) # 100 features
@pytest.mark.parametrize("cluster_settings", [
    ([6], (6,)), # 6 internal clusters
])
def test_get_parts(feature_size, cluster_settings):
    np.random.seed(0)

    gpu_clusters, cpu_clusters = cluster_settings
    feature = np.random.rand(feature_size)

    # GPU implementation
    parts_gpu = get_parts(feature, np.array(gpu_clusters, dtype=np.uint8)).get()

    # CPU implementation
    parts_cpu = get_parts_cpu(feature, cpu_clusters)

    print(f"\nTesting with feature_size={feature_size}, clusters={gpu_clusters}")
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
            print(f"\nDifferences found in partition {i}:")
            print(f"  Number of differing elements: {len(diff_indices)}")
            print(f"  First 10 differing indices: {diff_indices[:10]}")
            print(f"  GPU values at these indices: {parts_gpu[0][i][diff_indices[:10]]}")
            print(f"  CPU values at these indices: {parts_cpu[i][diff_indices[:10]]}")
            print(f"  Object values at these indices: {feature[diff_indices[:10]]}")

            # Verify partitions for differing elements
            for idx in diff_indices[:10]:
                expected_partition = verify_partition(feature, idx, n_clusters)
                assert parts_gpu[0][i][idx] == expected_partition, f"GPU partition mismatch for feature[{idx}]"

            assert False, f"GPU and CPU results don't match for {n_clusters} clusters"

    # Additional checks for multi-cluster settings
    if len(gpu_clusters) > 1:
        for i in range(len(gpu_clusters)):
            for j in range(i + 1, len(gpu_clusters)):
                if np.array_equal(parts_gpu[0][i], parts_cpu[j]):
                    print(f"\nUnexpected equality between partitions {i} and {j}:")
                    print(f"  Partition {i}: {parts_gpu[0][i]}")
                    print(f"  Partition {j}: {parts_cpu[j]}")
                    assert False, f"Partitions {i} and {j} should not be equal"