import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


def test_raw_kernel():
    # Define a raw kernel
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void my_raw_kernel(float* x, float* y, int n) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n) {
            y[tid] = x[tid] * x[tid];
        }
    }
    ''', 'my_raw_kernel')

    # Prepare input data
    n = 10
    x = cp.arange(n, dtype=cp.float32)

    # Allocate output array
    y = cp.empty_like(x)

    # Launch the kernel
    kernel((n,), (1,), (x, y, n))

    # Check the result
    assert cp.all(y == x * x)


def test_raw_kernel_with_thrust():
    N = 100
    code = """
    #include <thrust/count.h>
    #include <thrust/execution_policy.h>
    extern "C" __global__
    void xyzw_frequency_thrust_device(int *count, char *text, int n)
    {
      const char letters[] { 'x','y','z','w' };

      *count = thrust::count_if(thrust::device, text, text+n, [=](char c) {
        for (const auto x : letters) 
          if (c == x) return true;
        return false;
      });
    }"""
    kernel = cp.RawModule(code=code, backend='nvcc')
    code = kernel.get_function("xyzw_frequency_thrust_device")

    in_str = 'xxxzzzwwax'
    count = cp.zeros([1], dtype=cp.int64)
    in_arr = cp.array([ord(x) for x in in_str], dtype=cp.int8)

    # count[0] == 9 Define a raw kernel
    code(grid=(N,),block=(N,), args=(count, in_arr, len(in_str)))
    print()
    print(count)


def test_thrust_unique_count():
    N = 100
    code = """
    #include <thrust/unique.h>
    #include <thrust/execution_policy.h>
    extern "C" __global__
    void unique_count_thrust_device(int *count, int *data, int n)
    {
      *count = thrust::unique_count(thrust::device, data, data + n), thrust::equal_to<int>();
    }"""
    kernel = cp.RawModule(code=code, backend='nvcc')
    code = kernel.get_function("unique_count_thrust_device")

    # in_arr = cp.random.randint(0, 10, N)
    in_arr = cp.asarray([1, 3, 3, 3, 2, 2, 1], dtype=cp.int32)
    count = cp.zeros([1], dtype=cp.int32)

    # count[0] == 9 Define a raw kernel
    code(grid=(1,), block=(1,), args=(count, in_arr, 7))
    print(count)


def test_3d_raw_kernel():
    # Define a raw kernel to increment all elements by 1
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void increment_3d(float* array, int x, int y, int z) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int idz = blockIdx.z * blockDim.z + threadIdx.z;

        if (idx < x && idy < y && idz < z) {
            int index = idz * y * x + idy * x + idx;
            array[index] += 1.0f;
        }
    }
    ''', 'increment_3d')

    # Define the shape of the 3D array
    shape = (64, 64, 64)

    # Allocate and initialize a 3D array on the device
    d_array = cp.zeros(shape, dtype=cp.float32)

    # Define grid and block dimensions
    block_dim = (8, 8, 8)
    grid_dim = (
        (shape[0] + block_dim[0] - 1) // block_dim[0],
        (shape[1] + block_dim[1] - 1) // block_dim[1],
        (shape[2] + block_dim[2] - 1) // block_dim[2]
    )

    # Launch the kernel
    kernel(grid_dim, block_dim, (d_array, shape[0], shape[1], shape[2]))

    # Copy the result back to CPU for verification
    h_result = cp.asnumpy(d_array)

    # Verify the result
    expected = np.ones(shape, dtype=np.float32)
    np.testing.assert_array_almost_equal(h_result, expected, decimal=6)

    print("Test passed successfully!")


def test_3d_raw_kernel_1d_grid():
    # Define a raw kernel to increment all elements by 1 using 1D grid and block
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void increment_3d_1d(float* array, int x, int y, int z) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int total_size = x * y * z;

        if (tid < total_size) {
            int idz = tid / (x * y);
            int idy = (tid % (x * y)) / x;
            int idx = tid % x;

            array[tid] += 1.0f;
        }
    }
    ''', 'increment_3d_1d')

    # Define the shape of the 3D array
    shape = (64, 64, 64)

    # Allocate and initialize a 3D array on the device
    d_array = cp.zeros(shape, dtype=cp.float32)

    # Calculate total number of elements
    total_elements = np.prod(shape)

    # Define 1D grid and block dimensions
    block_dim = (256,)
    grid_dim = ((total_elements + block_dim[0] - 1) // block_dim[0],)

    # Launch the kernel
    kernel(grid_dim, block_dim, (d_array, shape[0], shape[1], shape[2]))

    # Copy the result back to CPU for verification
    h_result = cp.asnumpy(d_array)

    # Verify the result
    expected = np.ones(shape, dtype=np.float32)
    np.testing.assert_array_almost_equal(h_result, expected, decimal=6)

    print("Test passed successfully!")


def test_ravle():
    from sklearn.metrics import confusion_matrix
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    mat = confusion_matrix(y_true, y_pred)
    print(mat)


def test_3d_raw_kernel_grid_stride():
    # Define a raw kernel to increment all elements by 1 using grid-stride pattern
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void increment_3d_grid_stride(float* array, int total_size) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        for (int i = tid; i < total_size; i += blockDim.x * gridDim.x) {
            // Memory layout: CuPy, like NumPy, stores multi-dimensional arrays in contiguous memory
            // in row-major order (C-style). This means that elements are laid out sequentially in memory, 
            // regardless of the array's shape.
            array[i] += 1.0f;
        }
    }
    ''', 'increment_3d_grid_stride')

    # Define the shape of the 3D array
    shape = (64, 64, 64)

    # Allocate and initialize a 3D array on the device
    d_array = cp.zeros(shape, dtype=cp.float32)

    # Calculate total number of elements
    total_elements = np.prod(shape)

    # Define 1D grid and block dimensions
    block_dim = (256,)
    grid_dim = (min(1024, (total_elements + block_dim[0] - 1) // block_dim[0]),)

    # Launch the kernel
    kernel(grid_dim, block_dim, (d_array, total_elements))

    # Copy the result back to CPU for verification
    h_result = cp.asnumpy(d_array)

    # Verify the result
    expected = np.ones(shape, dtype=np.float32)
    np.testing.assert_array_almost_equal(h_result, expected, decimal=6)

    print("Test passed successfully!")


def test_3d_raw_kernel_grid_stride_indexing():
    # Define a raw kernel to increment all elements by 1 using grid-stride pattern
    # and explicit 3D indexing
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void increment_3d_grid_stride(float* array, int x, int y, int z) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int total_size = x * y * z;

        for (int i = tid; i < total_size; i += blockDim.x * gridDim.x) {
            int iz = i / (x * y);
            int iy = (i % (x * y)) / x;
            int ix = i % x;

            // Accessing the 3D array using 3D indices
            array[iz * (x * y) + iy * x + ix] += 1.0f;
        }
    }
    ''', 'increment_3d_grid_stride')

    # Define the shape of the 3D array
    shape = (64, 64, 64)

    # Allocate and initialize a 3D array on the device
    d_array = cp.zeros(shape, dtype=cp.float32)

    # Calculate total number of elements
    total_elements = np.prod(shape)

    # Define 1D grid and block dimensions
    block_dim = (256,)
    grid_dim = (min(1024, (total_elements + block_dim[0] - 1) // block_dim[0]),)

    # Launch the kernel
    kernel(grid_dim, block_dim, (d_array, shape[0], shape[1], shape[2]))

    # Copy the result back to CPU for verification
    h_result = cp.asnumpy(d_array)

    # Verify the result
    expected = np.ones(shape, dtype=np.float32)
    np.testing.assert_array_almost_equal(h_result, expected, decimal=6)

    print("Test passed successfully!")


def test_raft_api():
    code = cp.RawKernel(r'''
    extern "C" __global__
    #include <raft/core/handle.hpp>
    #include <raft/core/device_mdarray.hpp>
    #include <raft/random/make_blobs.cuh>
    #include <raft/distance/distance.cuh>

    raft::handle_t handle;

    int n_samples = 5000;
    int n_features = 50;

    auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int>(handle, n_samples);
    auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);

    raft::random::make_blobs(handle, input.view(), labels.view());

    auto metric = raft::distance::DistanceType::L2SqrtExpanded;
    raft::distance::pairwise_distance(handle, input.view(), input.view(), output.view(), metric);
    ''', 'raft_test')


def test_pair_wise_reduction():
    # Define a 3D parts array
    h_parts = np.array([
        [
            [1, 2, 3],
            [0, 2, 2],
            [1, 3, 3],
        ],
        [
            [1, 1, 1],
            [3, 1, 2],
            [1, 3, 3],
        ],
        [
            [0, 0, 3],
            [2, 1, 2],
            [1, 0, 1],
        ],
    ])
    # Host loop
    n_features = h_parts.shape[0]
    n_parts = h_parts.shape[1]
    n_objs = h_parts.shape[2]

    n_feat_comp = n_features * (n_features - 1) // 2

