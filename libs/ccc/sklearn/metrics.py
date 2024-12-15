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
from numba import njit, cuda
from cuda import grid

def sum_row(d_contingency, d_n_c, row, col):
    r = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if r < row:
        sum = 0
        for c in range(col):
            sum += d_contingency[r, c]
        d_n_c[r] = sum

def sum_col(d_contingency, d_values, row, col):
    c = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if c < col:
        sum = 0
        for r in range(row):
            sum += d_contingency[r, c]
        d_values[c] = sum

def sum_matrix_sq(d_contingency, d_sum_squares, row, col):
    c, r = grid(2)
    if r < row and c < col:
        cuda.atomic.add(d_sum_squares, 0, d_contingency[r, c]**2)

def dot_sum(d_contingency, d_n_k, d_sm, row, col):
    r = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if r < row:
        p_value = 0
        for c in range(col):
            p_value += d_contingency[r, c] * d_n_k[c]
        cuda.atomic.add(d_sm, 0, p_value)

def gpu_contingency_matrix(part0, part1, cont_mat):
    """
    GPU kernel to compute the contingency matrix.
    """
    idx = cuda.grid(1)
    if idx < part0.size:
        row = part0[idx]
        col = part1[idx]
        cuda.atomic.add(cont_mat, (row, col), 1)

@njit(cache=True, nogil=True)
def get_contingency_matrix(part0: np.ndarray, part1: np.ndarray) -> np.ndarray:
    """
    Given two clustering partitions with k0 and k1 number of clusters each, it
    returns a contingency matrix with k0 rows and k1 columns. It's an implementation of
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html,
    but the code is not based on their implementation.

    Args:
        part0: a 1d array with cluster assignments for n objects. It assumes that
            the cluster assignments are integers from 0 to k0-1.
        part1: a 1d array with cluster assignments for n objects. It assumes that
            the cluster assignments are integers from 0 to k1-1.

    Returns:
        A contingency matrix with k0 (number of clusters in part0) rows and k1
        (number of clusters in part1) columns. Each cell ij represents the
        number of objects grouped in cluster i (in part0) and cluster j (in
        part1).
    """
    k0, k1 = np.max(part0) + 1, np.max(part1) + 1
    cont_mat = np.zeros((k0, k1))

    # Allocate device memory
    d_part0 = cuda.to_device(part0)
    d_part1 = cuda.to_device(part1)
    d_cont_mat = cuda.to_device(cont_mat)

    # Launch the kernel
    threads_per_block = 256
    blocks_per_grid = (part0.size + threads_per_block - 1) // threads_per_block
    gpu_contingency_matrix[blocks_per_grid, threads_per_block](d_part0, d_part1, d_cont_mat)

    # Copy result back to host
    return d_cont_mat.copy_to_host()

@njit(cache=True, nogil=True)
def get_pair_confusion_matrix(part0: np.ndarray, part1: np.ndarray) -> np.ndarray:
    """
    Returns the pair confusion matrix from two clustering partitions. It is an
    implemenetation of
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html
    The code is based on the sklearn implementation. See copyright notice at the
    top of this file.

    Args:
        part0: a 1d array with cluster assignments for n objects.
        part1: a 1d array with cluster assignments for n objects.

    Returns:
        A pair confusion matrix with 2 rows and 2 columns. From sklearn's
        pair_confusion_matrix docstring: considering a pair of objects that is
        clustered together a positive pair, then as in binary classification the
        count of true negatives is in position 00, false negatives in 10, true
        positives in 11, and false positives in 01.
    """
    n_samples = np.int64(part0.shape[0])
    contingency = get_contingency_matrix(part0, part1)
    row, col = contingency.shape
    
    block_size = 128

    # Parallelize n_c = np.ravel(contingency.sum(axis=1)) -> sum all rows and store in 1d array
    n_c = np.zeros(row, dtype = np.int64)
    d_n_c = cuda.to_device(n_c)
    d_contingency = cuda.to_device(contingency)
    num_blocks_c = (row + block_size - 1) // block_size
    sum_row[num_blocks_c, block_size](d_contingency, d_n_c, row, col)
    cuda.deviceSynchronize()
    n_c = d_n_c.copy_to_host()

    # Parallelize n_k = np.ravel(contingency.sum(axis=0)) -> sum all cols and store in 1d array
    n_k = np.zeros(col, dtype = np.int64)
    d_n_k = cuda.to_device(n_k)
    num_blocks_k = (col + block_size - 1) // block_size
    sum_col[num_blocks_k, block_size](d_contingency, d_n_k, row, col)
    cuda.deviceSynchronize()
    n_k = d_n_k.copy_to_host()

    # Parallelize sum_squares = (contingency**2).sum() -> sum matrix^2
    sum_squares = np.zeros(1, dtype = np.int64)
    d_sum_squares = cuda.to_device(sum_squares)
    grid_dim = (num_blocks_k, num_blocks_c)
    block_dim = (16, 16)
    sum_matrix_sq[grid_dim, block_dim](d_contingency, d_sum_squares, row, col)
    cuda.deviceSynchronize()
    sum_squares = d_sum_squares.copy_to_host()[0]

    C = np.empty((2, 2), dtype = np.int64)
    C[1, 1] = sum_squares - n_samples

    # Parallelize C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    sm = np.zeros(1, dtype = np.int64)
    d_sm = cuda.to_device(sm)
    dot_sum[num_blocks_c, block_size](d_contingency, d_n_k, d_sm, row, col)
    cuda.deviceSynchronize()
    sm = d_sm.copy_to_host()[0]
    C[0, 1] = sm - sum_squares

    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares

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

