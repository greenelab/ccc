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
from numba import cuda
import math

@cuda.jit(device=True)
def find_unique(arr, max_unique):
    """Find unique elements in an array using shared memory."""
    unique = cuda.local.array(max_unique, dtype=np.int32)
    counts = cuda.local.array(max_unique, dtype=np.int32)
    num_unique = 0

    for i in range(len(arr)):
        found = False
        for j in range(num_unique):
            if arr[i] == unique[j]:
                counts[j] += 1
                found = True
                break
        if not found and num_unique < max_unique:
            unique[num_unique] = arr[i]
            counts[num_unique] = 1
            num_unique += 1

    return unique[:num_unique], counts[:num_unique], num_unique

@cuda.jit(device=True)
def compute_contingency_matrix(part0, part1, cont_mat, max_clusters):
    """Compute the contingency matrix using shared memory."""
    unique0, counts0, num_unique0 = find_unique(part0, max_clusters)
    unique1, counts1, num_unique1 = find_unique(part1, max_clusters)

    for i in range(num_unique0):
        for j in range(num_unique1):
            count = 0
            for k in range(len(part0)):
                if part0[k] == unique0[i] and part1[k] == unique1[j]:
                    count += 1
            cont_mat[i, j] = count

    return num_unique0, num_unique1

@cuda.jit(device=True)
def sum_2d_array(arr, rows, cols):
    """Sum elements in a 2D array."""
    total = 0
    for i in range(rows):
        for j in range(cols):
            total += arr[i, j]
    return total

@cuda.jit(device=True)
def sum_squares_2d_array(arr, rows, cols):
    """Sum squares of elements in a 2D array."""
    total = 0
    for i in range(rows):
        for j in range(cols):
            total += arr[i, j] * arr[i, j]
    return total

@cuda.jit(device=True)
def get_pair_confusion_matrix(part0, part1, max_clusters):
    """Compute the pair confusion matrix."""
    cont_mat = cuda.local.array((max_clusters, max_clusters), dtype=np.int32)
    num_clusters0, num_clusters1 = compute_contingency_matrix(part0, part1, cont_mat, max_clusters)

    n_samples = len(part0)
    sum_squares = sum_squares_2d_array(cont_mat, num_clusters0, num_clusters1)

    n_c = cuda.local.array(max_clusters, dtype=np.int32)
    n_k = cuda.local.array(max_clusters, dtype=np.int32)

    for i in range(num_clusters0):
        n_c[i] = sum(cont_mat[i, :num_clusters1])
    for j in range(num_clusters1):
        n_k[j] = sum(cont_mat[:num_clusters0, j])

    C = cuda.local.array((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = sum([cont_mat[i, j] * n_k[j] for i in range(num_clusters0) for j in range(num_clusters1)]) - sum_squares
    C[1, 0] = sum([cont_mat[i, j] * n_c[i] for i in range(num_clusters0) for j in range(num_clusters1)]) - sum_squares
    C[0, 0] = n_samples * n_samples - C[0, 1] - C[1, 0] - sum_squares

    return C

@cuda.jit(device=True)
def adjusted_rand_index(part0, part1, out, compare_pair_id, i, j, max_clusters):
    """
    Compute the adjusted Rand index (ARI) between two clustering partitions.
    """
    C = get_pair_confusion_matrix(part0, part1, max_clusters)
    tn, fp, fn, tp = C[0, 0], C[0, 1], C[1, 0], C[1, 1]

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        res = 1.0
    else:
        res = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))

    out[compare_pair_id, i, j] = res


# Main kernel function
# 1st iteration: try assign parts[i] (2D) to each block
@cuda.jit
def compute_ari(partitions, out, max_clusters):
    """
    CUDA kernel to compute ARI for multiple partition pairs.
    """
    compare_pair_id, i, j = cuda.grid(3)
    if compare_pair_id < partitions.shape[0] and i < partitions.shape[1] and j < partitions.shape[1]:
        part0 = partitions[compare_pair_id, i]
        part1 = partitions[compare_pair_id, j]
        adjusted_rand_index(part0, part1, out, compare_pair_id, i, j, max_clusters)