import cupy as cp
import numpy as np
from cupyx.jit import rawkernel
from cuml.metrics import adjusted_rand_score as cu_rnd_sc
from ccc.sklearn.metrics import adjusted_rand_index as ari
from numpy.typing import NDArray

# Assuming cu_rnd_sc is already defined as a device function
# If not, you'll need to implement it as a CUDA device function


@rawkernel()
def ari_kernel(x, y, res, m_x, m_y, n):
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < m_x * m_y:
        row_x = i // m_y
        row_y = i % m_y
        if x[row_x, 0] >= 0 and y[row_y, 0] >= 0:
            res[i] = cu_rnd_sc(x[row_x], y[row_y], n)
        else:
            res[i] = 0.0


def cdist_parts_cuda(x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
    """
    CUDA-accelerated version of cdist_parts_basic using CuPy.
    Each CUDA thread compares one row of x with one row of y.

    Args:
        x: a 2d array with m_x clustering partitions in rows and n objects in columns.
        y: a 2d array with m_y clustering partitions in rows and n objects in columns.

    Returns:
        A 2d array with m_x rows and m_y columns and the ARI between each partition pair.
    """
    m_x, n = x.shape
    m_y, _ = y.shape
    res = cp.zeros(m_x * m_y, dtype=cp.float32)

    threads_per_block = 256
    blocks = (m_x * m_y + threads_per_block - 1) // threads_per_block

    ari_kernel[blocks, threads_per_block](x, y, res, m_x, m_y, n)

    return res.reshape(m_x, m_y)


def cdist_parts_basic(x: NDArray, y: NDArray) -> NDArray[float]:
    """
    It implements the same functionality in scipy.spatial.distance.cdist but
    for clustering partitions, and instead of a distance it returns the adjusted
    Rand index (ARI). In other words, it mimics this function call:

        cdist(x, y, metric=ari)

    Only partitions with positive labels (> 0) are compared. This means that
    partitions marked as "singleton" or "empty" (categorical data) are not
    compared. This has the effect of leaving an ARI of 0.0 (zero).

    Args:
        x: a 2d array with m_x clustering partitions in rows and n objects in
          columns.
        y: a 2d array with m_y clustering partitions in rows and n objects in
          columns.

    Returns:
        A 2d array with m_x rows and m_y columns and the ARI between each
        partition pair. Each ij entry is equal to ari(x[i], y[j]) for each i
        and j.
    """
    res = np.zeros((x.shape[0], y.shape[0]))

    for i in range(res.shape[0]):
        if x[i, 0] < 0:
            continue

        for j in range(res.shape[1]):
            if y[j, 0] < 0:
                continue

            res[i, j] = ari(x[i], y[j])

    return res


# Test function
def test_cdist_parts_cuda():
    # Generate sample data
    np.random.seed(0)
    m_x, m_y, n = 100, 80, 1000
    x = np.random.randint(0, 5, size=(m_x, n))
    y = np.random.randint(0, 5, size=(m_y, n))

    # Convert to CuPy arrays
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)

    # Run CUDA version
    res_cuda = cdist_parts_cuda(x_gpu, y_gpu)

    # Run CPU version for comparison
    res_cpu = cdist_parts_basic(x, y)

    # Compare results
    cp.cuda.Stream.null.synchronize()
    res_cuda_np = cp.asnumpy(res_cuda)

    assert np.allclose(res_cuda_np, res_cpu, atol=1e-6), "CUDA and CPU results do not match"

    print("CUDA implementation matches CPU implementation")

    # Performance comparison
    import time

    start_time = time.time()
    for _ in range(10):
        cdist_parts_cuda(x_gpu, y_gpu)
    cp.cuda.Stream.null.synchronize()
    cuda_time = (time.time() - start_time) / 10

    start_time = time.time()
    for _ in range(10):
        cdist_parts_basic(x, y)
    cpu_time = (time.time() - start_time) / 10

    print(f"CUDA time: {cuda_time:.6f} seconds")
    print(f"CPU time: {cpu_time:.6f} seconds")
    print(f"Speedup: {cpu_time / cuda_time:.2f}x")


from cupyx import jit


@jit.rawkernel()
def elementwise_copy(x, y, size):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        y[i] = x[i]


def test_elementwise():
    size = cp.uint32(2 ** 22)
    x = cp.random.normal(size=(size,), dtype=cp.float32)
    y = cp.empty((size,), dtype=cp.float32)

    elementwise_copy((128,), (1024,), (x, y, size))  # RawKernel style


    assert (x == y).all()

    elementwise_copy[128, 1024](x, y, size)  #  Numba style
    assert (x == y).all()