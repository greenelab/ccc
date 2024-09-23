import numpy as np
import cupy as cp
from numba import njit
from numba import cuda
import rmm


def adjusted_rand_index(
                        part0: np.ndarray,
                        part1: np.ndarray,
                        size: int,
                        out: np.ndarray,
                        compare_pair_id: int,
                        i: int,
                        j: int,
                        stream: cp.cuda.Stream = None):
    """
    Computes the adjusted Rand index (ARI) between two clustering partitions.
    The code is based on the sklearn implementation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    See copyright notice at the top of this file.

    Host function to coordinate the GPU kernel.

    Args:
        part0: a 1d array with cluster assignments for n objects.
        part1: a 1d array with cluster assignments for n objects.
        size: the number of objects in the partitions.
        out: pointer to the output array containing all the ARI values. # TODO: make local
        compare_pair_id: the index of the pair of partitions to compare.
        i: the index of the first partition.
        j: the index of the second partition.
        stream: the CUDA stream to use.

    Returns:
        A number representing the adjusted Rand index between two clustering
        partitions. This number is between something around 0 (partitions do not
        match; it could be negative in some cases) and 1.0 (perfect match).
    """
    # TODO:
    # Implement numpy ravel in the kernel using shared memory?
    # Use different streams for different pairs?
    # Ref api: CUML confusion_matrix
    if not size >= 2:
        raise ValueError("Need at least two samples to compare.")



    (tn, fp), (fn, tp) = get_pair_confusion_matrix(part0, part1)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        res = 1.0

    res = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    out[compare_pair_id, i, j] = res


def ari_dim2(parts: cp.ndarray, n_features_comp, out: cp.ndarray):
    """
    Function to compute the ARI between partitions on the GPU. This function is responsible for launching the kernel
    in different streams for each pair of partitions.

    Args:
        parts: 3D device array with cluster assignments for x features, y partitions, and z objects.
        Example initialization for this array: d_parts = cp.empty((nx, ny, nz), dtype=np.int16) - 1

        n_features_comp: Pre-computed number of features to compare.

        out: Pointer to the pre-allocated 1D device output array with length of n_features_comp.
    """

    # Can use non-blocking CPU scheduling or CUDA dynamic parallelism to launch the kernel for each pair of partitions.

    raise NotImplementedError("Not implemented yet")
