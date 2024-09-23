import cupy as cp
import numpy as np
from ccc.sklearn.metrics_gpu2 import device_func_str
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
    cuda_code = device_func_str + test_kernel_code
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
