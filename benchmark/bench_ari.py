import pytest
import time
import cupy as cp
import numpy as np
from ccc.sklearn.metrics_gpu2 import (
    d_get_confusion_matrix_str,
    d_get_coords_from_index_str,
    d_unravel_index_str,
    d_get_contingency_matrix_str,
    k_ari_str,
)
from ccc.sklearn.metrics import (
    adjusted_rand_index,
)


def generate_pairwise_combinations(arr):
    pairs = []
    num_slices = arr.shape[0]  # Number of 2D arrays in the 3D array

    for i in range(num_slices):
        for j in range(i + 1, num_slices):  # Only consider pairs in different slices
            for row_i in arr[i]:  # Each row in slice i
                for row_j in arr[j]:  # Pairs with each row in slice j
                    pairs.append([row_i, row_j])

    # Convert list of pairs to a NumPy array
    return np.array(pairs)


@pytest.mark.parametrize("n_features, n_parts, n_objs, k", [
    (100, 10, 300, 10),
    (100, 20, 300, 10),
    # (100, 20, 1000, 10), # wrong results
    # (200, 20, 300, 10), # illegal mem access
    # (1000, 10, 300, 10), # out of gpu mem
])
@pytest.mark.parametrize("block_size", [1024])
def test_pairwise_ari(n_features, n_parts, n_objs, k, block_size):
    parts = np.random.randint(0, k, size=(n_features, n_parts, n_objs), dtype=np.int32)
    # Create test inputs
    n_features, n_parts, n_objs = parts.shape
    n_feature_comp = n_features * (n_features - 1) // 2
    n_aris = n_feature_comp * n_parts * n_parts
    ref_aris = np.zeros(n_aris, dtype=np.float32)
    # Get partition pairs
    pairs = generate_pairwise_combinations(parts)

    start = time.time()
    # Use map-reduce to compute ARIs for all pairs of partitions
    for i, (part0, part1) in enumerate(pairs):
        ari = adjusted_rand_index(part0, part1)
        ref_aris[i] = ari
    end = time.time()
    time_cpu = end - start
    print(f"\nFor {n_features} features, {n_parts} partitions, {n_objs} objects:")
    print(f"CPU Time taken: {time_cpu:.4f} seconds")

    # Compute ARIs using the CUDA kernel
    grid_size = n_aris
    s_mem_size = n_objs * 2 * cp.int32().itemsize  # For the partition pair to be compared
    s_mem_size += 2 * k * cp.int32().itemsize  # For the internal sum arrays
    s_mem_size += 4 * cp.int32().itemsize  # For the 2 x 2 confusion matrix

    start = time.time()
    d_out = cp.empty(n_aris, dtype=cp.float32)
    d_parts = cp.asarray(parts, dtype=cp.int32)
    d_parts_pairs = cp.empty((n_aris, 2, n_objs), dtype=cp.int32)
    # Each pair of partitions will be compared, used for debugging purposes

    # Compile the CUDA kernel
    kernel_code = d_unravel_index_str + d_get_coords_from_index_str + d_get_contingency_matrix_str + d_get_confusion_matrix_str + k_ari_str
    module = cp.RawModule(code=kernel_code, backend='nvcc')
    kernel = module.get_function("ari")
    # Launch the kernel
    kernel((grid_size,), (block_size,), (d_parts,
                                                        n_aris,
                                                        n_features,
                                                        n_parts,
                                                        n_objs,
                                                        n_parts * n_objs,
                                                        n_parts * n_parts,
                                                        k,
                                                        d_out,
                                                        d_parts_pairs),
                                                        shared_mem=s_mem_size)
    end = time.time()
    time_gpu = end - start
    print(f"GPU Time taken: {time_gpu:.4f} seconds")
    cp.cuda.runtime.deviceSynchronize()
    # Get results back to host
    h_out = cp.asnumpy(d_out)
    assert np.allclose(h_out, ref_aris)
