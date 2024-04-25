import numpy as np
import pandas as pd
from numba import cuda

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
def get_contingency_matrix(random_feature1_device , random_feature2_device, part0_unique_device, part1_unique_device, cont_mat_device, part1_k_device, part1_j_device, part0_i_device):
    """
    Given two clustering partitions with k0 and k1 number of clusters each, it
    returns a contingency matrix with k0 rows and k1 columns. It's an implementation of
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html,
    but the code is not based on their implementation.
    Args:
        part0: a 1d array with cluster assignments for n objects.
        part1: a 1d array with cluster assignments for n objects.
    Returns:
        A contingency matrix with k0 (number of clusters in part0) rows and k1
        (number of clusters in part1) columns. Each cell ij represents the
        number of objects grouped in cluster i (in part0) and cluster j (in
        part1).
    """
    
    #Creating the grid
    #x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh




    #part0_unique = np.unique(array1)
    #part1_unique = np.unique(array2)
    #cont_mat = np.zeros((len(part0_unique), len(part1_unique)))
    
    if i < M:
        part0_k_device = part0_unique_device[i]
        if j < N:
            part1_k_device = part1_unique_device[j]
            #cuda.atomic.compare_and_swap_element(part0_i_device , 
            part0_i_device = random_feature1_device == part0_k_device
            part1_j_device = random_feature2_device == part1_k_device
            cont_mat_device[i, j] = np.sum(part0_i_device & part1_j_device)
    
    return cont_mat_device

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

    # Computation using the contingency data
    contingency = get_contingency_matrix(part0, part1)
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency**2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
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


if __name__ == '__main__':
   
    # Arrays
    random_feature1 = np.random.rand(1000).astype('f')
    random_feature2 = np.random.rand(1000).astype('f')
    
    # Processing the unique arrays:
    part0_unique = np.unique(random_feature1)
    part1_unique = np.unique(random_feature2)
    cont_mat = np.zeros((len(part0_unique), len(part1_unique)))
    part1_k = np.ones(1, dtype=np.float64) 
    part1_j =  np.ones(1, dtype=np.float64)
    part0_i =  np.ones(1, dtype=np.float64)
    # Getting other important parts of for the GPU setting:
    threadsperblock = (128, 128)
    M = part0_unique.shape[0]
    N = part1_unique.shape[0]
    blockspergrid_x = M + (threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = N + (threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    #Senign them to the GPU:
    random_feature1_device = cuda.to_device(random_feature1)
    random_feature2_device = cuda.to_device(random_feature2)
    part0_unique_device = cuda.to_device(part0_unique)
    part1_unique_device = cuda.to_device(part1_unique)
    cont_mat_device = cuda.to_device(cont_mat)
    part1_k_device = cuda.to_device(part1_k)
    part1_j_device = cuda.to_device(part1_j)
    part0_i_device = cuda.to_device(part0_i)
    print("checkpoint")
    # Calling the get_contingency
    out_device = get_contingency_matrix[blockspergrid, threadsperblock](random_feature1_device , random_feature2_device, part0_unique_device, part1_unique_device, cont_mat_device, part1_k_device, part1_j_device, part0_i_device)
    print(out_device)


