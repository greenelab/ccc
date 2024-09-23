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
    kernel = cp.RawModule(code=code,backend='nvcc')
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
