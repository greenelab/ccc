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


@njit(cache=True, nogil=True)
def get_contingency_matrix(part0: np.ndarray, part1: np.ndarray) -> np.ndarray:
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
    part0_unique = np.unique(part0)
    part1_unique = np.unique(part1)

    cont_mat = np.zeros((len(part0_unique), len(part1_unique)))

    for i in range(len(part0_unique)):
        part0_k = part0_unique[i]

        for j in range(len(part1_unique)):
            part1_k = part1_unique[j]

            part0_i = part0 == part0_k
            part1_j = part1 == part1_k

            cont_mat[i, j] = np.sum(part0_i & part1_j)

    return cont_mat


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
