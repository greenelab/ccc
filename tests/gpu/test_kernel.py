import pytest
import math
import cupy as cp
import numpy as np
from ccc.sklearn.metrics_gpu2 import (
    d_get_coords_from_index_str,
    d_unravel_index_str,
    d_get_contingency_matrix_str,
    k_ari_str
)
from ccc.coef import get_coords_from_index
from ccc.sklearn.metrics import get_contingency_matrix


def test_get_coords_from_index_kernel():
    test_kernel_code = """
    extern "C" __global__
    void test_kernel(int n_obj, int* indices, int* results, int num_indices) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < num_indices) {
            int x, y;
            get_coords_from_index(n_obj, indices[tid], &x, &y);
            results[tid * 2] = x;
            results[tid * 2 + 1] = y;
        }
    }
    """
    cuda_code = d_get_coords_from_index_str + test_kernel_code
    module = cp.RawModule(code=cuda_code, backend='nvcc')
    kernel = module.get_function("test_kernel")

    # Test parameters
    n_obj = 10
    num_indices = 45  # (n_obj * (n_obj - 1)) // 2

    # Create input indices
    indices = cp.arange(num_indices, dtype=cp.int32)

    # Allocate memory for results
    d_results = cp.empty(num_indices * 2, dtype=cp.int32)

    # Launch the kernel
    threads_per_block = 256
    blocks = (num_indices + threads_per_block - 1) // threads_per_block
    kernel((blocks,), (threads_per_block,), (n_obj, indices, d_results, num_indices))

    # Get results back to host
    h_results = cp.asnumpy(d_results)

    # Compare with Python implementation
    for i in range(num_indices):
        x_cuda, y_cuda = h_results[i * 2], h_results[i * 2 + 1]
        x_py, y_py = get_coords_from_index(n_obj, i)

        assert x_cuda == x_py, f"Mismatch in x for index {i}: CUDA={x_cuda}, Python={x_py}"
        assert y_cuda == y_py, f"Mismatch in y for index {i}: CUDA={y_cuda}, Python={y_py}"

    print("All tests passed successfully!")


@pytest.mark.parametrize("num_cols, num_indices", [
    (10, 45),
    (15, 100),
    (20, 200)
])
def test_unravel_index_kernel(num_cols, num_indices):
    test_kernel_code = """
    extern "C" __global__ void test_unravel_index_kernel(size_t* flat_indices, size_t* rows, size_t* cols, size_t num_cols, size_t num_indices) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < num_indices) {
            unravel_index(flat_indices[tid], num_cols, &rows[tid], &cols[tid]);
        }
    }
    """

    cuda_code = d_unravel_index_str + test_kernel_code
    # Compile the CUDA kernel
    module = cp.RawModule(code=cuda_code, backend='nvcc')
    kernel = module.get_function("test_unravel_index_kernel")

    # Create test inputs
    flat_indices = cp.arange(num_indices, dtype=cp.uint64)

    # Allocate memory for results (rows and cols)
    d_rows = cp.empty(num_indices, dtype=cp.uint64)
    d_cols = cp.empty(num_indices, dtype=cp.uint64)

    # Launch the kernel
    threads_per_block = 256
    blocks = (num_indices + threads_per_block - 1) // threads_per_block
    kernel((blocks,), (threads_per_block,), (flat_indices, d_rows, d_cols, num_cols, num_indices))

    # Get results back to host
    h_rows = cp.asnumpy(d_rows)
    h_cols = cp.asnumpy(d_cols)

    # Compare with NumPy's unravel_index implementation
    for i in range(num_indices):
        # Use numpy.unravel_index as the reference
        # row_py, col_py = divmod(i, num_cols)
        row_py, col_py = np.unravel_index(i, (num_cols, num_cols))
        row_cuda, col_cuda = h_rows[i], h_cols[i]

        # Assertions to ensure CUDA and NumPy match
        assert row_cuda == row_py, f"Mismatch in row for index {i}: CUDA={row_cuda}, NumPy={row_py}"
        assert col_cuda == col_py, f"Mismatch in col for index {i}: CUDA={col_cuda}, NumPy={col_py}"

    print("All tests passed successfully!")


@pytest.mark.parametrize("n", [100, 1000, 10000])
@pytest.mark.parametrize("threads_per_block", [1, 2, 64, 128, 256, 512])
@pytest.mark.parametrize("k", [2, 5, 10])
def test_get_contingency_matrix_kernel(n, threads_per_block, k):
    test_kernel_code = """
    extern "C"
    __global__ void test_kernel(int* part0, int* part1, int n, int* cont_mat, int k) {
        extern __shared__ int shared_cont_mat[];
        
        // Call the function to compute contingency matrix in shared memory
        get_contingency_matrix(part0, part1, n, shared_cont_mat, k);
        
        // Copy shared memory back to global memory
        int tid = threadIdx.x;
        int num_threads = blockDim.x;
        
        for (int i = tid; i < k * k; i += num_threads) {
            atomicAdd(&cont_mat[i], shared_cont_mat[i]);
        }
    }
    """
    cuda_code = d_get_contingency_matrix_str + test_kernel_code
    # Compile the CUDA kernel
    module = cp.RawModule(code=cuda_code, backend='nvcc')
    kernel = module.get_function("test_kernel")

    # Generate random partitions
    part0 = np.random.randint(0, k, size=n, dtype=np.int32)
    part1 = np.random.randint(0, k, size=n, dtype=np.int32)

    # Transfer data to GPU
    d_part0 = cp.asarray(part0)
    d_part1 = cp.asarray(part1)
    d_cont_mat = cp.zeros((k, k), dtype=cp.int32)

    # Launch the kernel
    blocks = 1  # Each pair of partitions is handled by only one block (to fully utilize shared memory)
    shared_mem_size = k * k * 4  # 4 bytes per int
    kernel((blocks,), (threads_per_block,),
           (d_part0, d_part1, n, d_cont_mat, k),
           shared_mem=shared_mem_size)

    # Get results back to host
    h_cont_mat = cp.asnumpy(d_cont_mat)

    # Compare with reference implementation
    ref_cont_mat = get_contingency_matrix(part0, part1)

    np.testing.assert_array_equal(h_cont_mat, ref_cont_mat,
                                  err_msg=f"CUDA and reference implementations do not match for n={n}, threads_per_block={threads_per_block}, k={k}")
    print(f"Test passed successfully for n={n}, threads_per_block={threads_per_block}, k={k}")


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


@pytest.mark.parametrize("parts", [
    # 3D array
    np.array([
        [[0, 1, 2, 3],
         [0, 2, 3, 4],
         [0, 3, 4, 5]],

        [[1, 1, 2, 3],
         [1, 2, 3, 4],
         [1, 3, 4, 5]],

        [[2, 1, 2, 3],
         [2, 2, 3, 4],
         [2, 3, 4, 5]]
    ])
])
def test_art_parts_selection(parts):
    pairs = generate_pairwise_combinations(parts)

    kernel_code = d_unravel_index_str + d_get_coords_from_index_str + k_ari_str
    # Compile the CUDA kernel
    module = cp.RawModule(code=kernel_code, backend='nvcc')
    kernel = module.get_function("ari")

    # Create test inputs
    n_features, n_parts, n_objs = parts.shape
    n_feature_comp = n_features * (n_features - 1) // 2
    n_aris = n_feature_comp * n_parts * n_parts
    block_size = 2
    grid_size = (n_aris + block_size - 1) // block_size
    s_mem_size = n_objs * 2 * cp.int32().itemsize

    d_out = cp.empty(n_aris, dtype=cp.int32)
    d_parts = cp.asarray(parts, dtype=cp.int32)
    # Each pair of partitions will be compared, used for debugging purposes
    d_parts_pairs = cp.empty((n_aris, 2, n_objs), dtype=cp.int32)
    d_uniqs = cp.empty(n_objs, dtype=cp.int32)

    # Print stats
    print(f"Number of ARIs: {n_aris}")
    # Print kernel configuration
    print(f"Grid size: {grid_size}, Block size: {block_size}, Shared memory: {s_mem_size}")
    # Launch the kernel
    kernel((grid_size,), (block_size,), (d_parts,
                                                        d_uniqs,
                                                        n_aris,
                                                        n_features,
                                                        n_parts,
                                                        n_objs,
                                                        n_parts * n_objs,
                                                        n_objs * n_objs,
                                                        d_out,
                                                        d_parts_pairs),
                                                        shared_mem=s_mem_size)
    cp.cuda.runtime.deviceSynchronize()
    # Get results back to host
    h_parts_pairs = cp.asnumpy(d_parts_pairs)
    print(h_parts_pairs)
    # Assert pairs == d_parts_pairs
    assert np.all(np.equal(h_parts_pairs, pairs))
