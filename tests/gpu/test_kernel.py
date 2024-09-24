import pytest
import math
import cupy as cp
import numpy as np
from ccc.sklearn.metrics_gpu2 import d_get_coords_from_index_str, d_unravel_index_str
from ccc.coef import get_coords_from_index


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

