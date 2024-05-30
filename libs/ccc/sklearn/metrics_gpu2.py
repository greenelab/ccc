"""
Contains implementations of different metrics in sklearn but optimized for numba.

Some code (indicated in each function) is based on scikit-learn's code base
(https://github.com/scikit-learn), for which the copyright notice and license
are shown below.

BSD 3-Clause License

Copyright (c) 2007-2021 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from numba import njit
from numba import cuda


@cuda.jit
def compute_sum_squares(contingency, result):
    """
    CUDA kernel to compute the sum of squares of the contingency matrix elements.

    Args:
        contingency: The contingency matrix.
        result: The output array to store the sum of squares.
    """
    i, j = cuda.grid(2)

    if i < contingency.shape[0] and j < contingency.shape[1]:
        cuda.atomic.add(result, 0, contingency[i, j] ** 2)


def get_pair_confusion_matrix(part0: np.ndarray, part1: np.ndarray) -> np.ndarray:
    """
    Returns the pair confusion matrix from two clustering partitions using CUDA.

    Args:
        part0: A 1D array with cluster assignments for n objects.
        part1: A 1D array with cluster assignments for n objects.

    Returns:
        A pair confusion matrix with 2 rows and 2 columns.
    """
    n_samples = np.int64(part0.shape[0])

    # Compute the contingency matrix
    contingency = get_contingency_matrix(part0, part1)

    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))

    # Allocate space for the sum of squares result
    sum_squares = np.zeros(1, dtype=np.int64)

    # Define the number of threads per block and the number of blocks per grid
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(contingency.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(contingency.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch the CUDA kernel to compute the sum of squares
    compute_sum_squares[blockspergrid, threadsperblock](contingency, sum_squares)

    sum_squares = sum_squares[0]

    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = np.dot(contingency, n_k).sum() - sum_squares
    C[1, 0] = np.dot(contingency.T, n_c).sum() - sum_squares
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares

    return C


def adjusted_rand_index(part0: np.ndarray, part1: np.ndarray) -> float:
    """
    Computes the adjusted Rand index (ARI) between two clustering partitions.
    The code is based on the sklearn implementation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    See copyright notice at the top of this file.

    This function should not be compiled with numba, since it depends on
    arbitrarily large interger variable (supported by Python) to correctly
    compute the ARI in large partitions.

    Args:
        part0: a 1d array with cluster assignments for n objects.
        part1: a 1d array with cluster assignments for n objects.

    Returns:
        A number representing the adjusted Rand index between two clustering
        partitions. This number is between something around 0 (partitions do not
        match; it could be negative in some cases) and 1.0 (perfect match).
    """
    (tn, fp), (fn, tp) = get_pair_confusion_matrix(part0, part1)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


@cuda.jit
def compute_contingency_matrix(part0, part1, part0_unique, part1_unique, cont_mat):
    """
    CUDA kernel to compute the contingency matrix.

    Args:
        part0: 1D array with cluster assignments for n objects.
        part1: 1D array with cluster assignments for n objects.
        part0_unique: Unique cluster labels in part0.
        part1_unique: Unique cluster labels in part1.
        cont_mat: The output contingency matrix.

    Each thread computes a single element of the contingency matrix.
    """
    i, j = cuda.grid(2)  # Get the thread indices in the grid

    # Check if the thread indices are within the bounds of the unique clusters
    if i < len(part0_unique) and j < len(part1_unique):
        part0_k = part0_unique[i]  # Cluster label in part0
        part1_k = part1_unique[j]  # Cluster label in part1

        count = 0  # Initialize the count for this element
        for idx in range(len(part0)):
            # Count the number of objects in both clusters i and j
            if part0[idx] == part0_k and part1[idx] == part1_k:
                count += 1
        cont_mat[i, j] = count  # Store the result in the contingency matrix


def get_contingency_matrix(part0: np.ndarray, part1: np.ndarray) -> np.ndarray:
    """
    Compute the contingency matrix for two clustering partitions using CUDA.

    Args:
        part0: 1D array with cluster assignments for n objects.
        part1: 1D array with cluster assignments for n objects.

    Returns:
        A contingency matrix with k0 rows and k1 columns, where k0 is the number
        of clusters in part0 and k1 is the number of clusters in part1. Each cell
        (i, j) represents the number of objects in cluster i (part0) and cluster j (part1).
    """
    part0_unique = np.unique(part0)  # Find unique clusters in part0
    part1_unique = np.unique(part1)  # Find unique clusters in part1

    cont_mat = np.zeros((len(part0_unique), len(part1_unique)), dtype=np.int32)  # Initialize the contingency matrix

    # Define the number of threads per block and the number of blocks per grid
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(len(part0_unique) / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(len(part1_unique) / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch the CUDA kernel to compute the contingency matrix
    compute_contingency_matrix[blockspergrid, threadsperblock](part0, part1, part0_unique, part1_unique, cont_mat)

    return cont_mat


@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1

def _test_cuda1():
    # Initialize the array
    data = np.ones(64)
    print(f"Data before kernel call: {data}")
    # Set the number of threads in a block
    threads_per_block = 32
    # Calculate the number of thread blocks in the grid
    blocks_per_grid = (data.size + (threads_per_block - 1)) // threads_per_block
    # Call the kernel
    increment_by_one[blocks_per_grid, threads_per_block](data)
    print(f"Data after kernel call: {data}")
    return


def _test_ari():
    part0 = np.array([0, 0, 1, 1, 2, 2])
    part1 = np.array([0, 0, 1, 1, 2, 2])
    print(adjusted_rand_index(part0, part1))  # 1.0

    part0 = np.array([0, 0, 1, 1])
    part1 = np.array([0, 0, 1, 2])
    print(adjusted_rand_index(part0, part1))  # 0.57

    part0 = np.array([0, 0, 1, 1])
    part1 = np.array([0, 1, 0, 1])
    print(adjusted_rand_index(part0, part1))  # -0.5


def print_device_info():
    # Get the current device
    device = cuda.get_current_device()
    print(dir(device))
    # Print device information
    print("Device Information:")
    print(f"Device ID: {device.id}")
    print(f"Name: {device.name}")
    # print(f"Total Memory: {device.total_memory / (1024 ** 3):.2f} GB")
    print(f"Multiprocessor Count: {device.MULTIPROCESSOR_COUNT}")
    print(f"Max Threads per Block: {device.MAX_THREADS_PER_BLOCK}")
    # print(f"Max Threads per Multiprocessor: {device.MAX_THREADS_PER_MULTIPROCESSOR}")
    print(f"Max Block Dim X: {device.MAX_BLOCK_DIM_X}")
    print(f"Max Block Dim Y: {device.MAX_BLOCK_DIM_Y}")
    print(f"Max Block Dim Z: {device.MAX_BLOCK_DIM_Z}")
    print(f"Max Grid Dim X: {device.MAX_GRID_DIM_X}")
    print(f"Max Grid Dim Y: {device.MAX_GRID_DIM_Y}")
    print(f"Max Grid Dim Z: {device.MAX_GRID_DIM_Z}")
    print(f"Warp Size: {device.WARP_SIZE}")
    print(f"Compute Capability: {device.compute_capability}")
    print(f"Concurrent Kernels: {device.CONCURRENT_KERNELS}")
    print(f"PCI Bus ID: {device.PCI_BUS_ID}")
    print(f"PCI Device ID: {device.PCI_DEVICE_ID}")
    print(f"PCI Domain ID: {device.PCI_DOMAIN_ID}")


if __name__ == '__main__':
    part0 = np.array([0, 0, 1, 1, 2, 2])
    part1 = np.array([1, 0, 2, 1, 0, 2])
    cont_matrix = get_contingency_matrix(part0, part1)
    print(cont_matrix)

    _test_ari()
