import numpy as np
from numba import njit


@njit(cache=True)
def get_contingency_matrix(part0, part1):
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


@njit(cache=True)
def get_pair_confusion_matrix(part0, part1):
    """
    TODO: taken from sklearn pair_confusion_matrix
    """
    n_samples = np.int64(part0.shape[0])

    # Computation using the contingency data
    contingency = get_contingency_matrix(
        part0, part1
    )
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency ** 2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares
    return C


@njit(cache=True)
def adjusted_rand_index(part0, part1):
    """
    TODO: taken from sklearn
    """
    (tn, fp), (fn, tp) = get_pair_confusion_matrix(part0, part1)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))
