import numpy as np
from numba import njit
from numba import cuda


def adjusted_rand_index(part0: np.ndarray, part1: np.ndarray, out: np.ndarray, compare_pair_id: int, i: int,
                        j: int) -> float:
    """
    Computes the adjusted Rand index (ARI) between two clustering partitions.
    The code is based on the sklearn implementation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    See copyright notice at the top of this file.

    Host function to coordinate the GPU kernel.

    Args:
        part0: a 1d array with cluster assignments for n objects.
        part1: a 1d array with cluster assignments for n objects.
        out: pointer to the output array containing all the ARI values. # TODO: make local

    Returns:
        A number representing the adjusted Rand index between two clustering
        partitions. This number is between something around 0 (partitions do not
        match; it could be negative in some cases) and 1.0 (perfect match).
    """
    # TODO:
    # Implement numpy ravel in the kernel using shared memory?

    (tn, fp), (fn, tp) = get_pair_confusion_matrix(part0, part1)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        res = 1.0

    res = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    out[compare_pair_id, i, j] = res
