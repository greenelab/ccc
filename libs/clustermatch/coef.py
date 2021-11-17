"""
Contains function that implement the Clustermatch coefficient
(https://doi.org/10.1093/bioinformatics/bty899).
"""
from typing import Iterable

import numpy as np
from numba import njit, prange
from numba.typed import List

from clustermatch.metrics import adjusted_rand_index as ari


@njit(cache=True)
def _get_perc_from_k(k: int) -> list[float]:
    """
    It returns the percentiles (from 0.0 to 1.0) that separate the data into k
    clusters. For example, if k=2, it returns [0.5]; if k=4, it returns [0.25,
    0.50, 0.75].

    Args:
        k: number of clusters. If less than 2, the function returns an empty
            list.

    Returns:
        A list of percentiles (from 0.0 to 1.0).
    """
    return [(1.0 / k) * i for i in range(1, k)]


@njit(cache=True)
def rank(data: np.ndarray) -> np.ndarray:
    """
    It returns the ranks of a numpy array. It's an implementation of
    scipy.stats.rankdata (method="average") that can be compiled by numba.
    Ranks start with 1.

    Args:
        data: a 1d array with numeric data.

    Returns:
        A 1d array with the ranks of the input data.
    """
    data_sorted_idx = data.argsort()
    data_sorted = data[data_sorted_idx]

    data_ranks = data_sorted_idx.argsort().astype(np.float64)

    # handle ties with the average
    first_idx = data_sorted_idx[0]
    first_rank = data_ranks[first_idx]
    current_rank_group_idxs = [first_idx]
    current_rank_group = [first_rank]

    for i in range(1, data.shape[0]):
        current_idx = data_sorted_idx[i]
        current_rank = data_ranks[current_idx]

        if data_sorted[i] == data_sorted[i - 1]:
            current_rank_group_idxs.append(current_idx)
            current_rank_group.append(current_rank)

            if i < (data.shape[0] - 1):
                continue

        if len(current_rank_group) > 1:
            assert len(current_rank_group) == len(set(current_rank_group))
            assert len(current_rank_group_idxs) == len(set(current_rank_group_idxs))

            avg_rank = np.array(current_rank_group).mean()
            data_ranks[np.array(current_rank_group_idxs)] = avg_rank

        current_rank_group = [current_rank]
        current_rank_group_idxs = [current_idx]

    return data_ranks + 1


@njit(cache=True)
def run_quantile_clustering(data: np.ndarray, k: int) -> np.ndarray:
    """
    Performs a simple quantile clustering on one dimensional data (1d). Quantile
    clustering is defined as the procedure that forms clusters in 1d data by
    separating objects using quantiles (for instance, if the median is used, two
    clusters are generated with objects separated by the median). In the case
    data contains all the same values (zero variance), this implementation can
    return less clusters than specified with k.

    Args:
        data: a 1d numpy array with numerical values.
        k: the number of clusters to split the data into.

    Returns:
        A 1d array with the data partition.
    """
    data_perc = rank(data) / len(data)
    data_perc_sort_idx = data_perc.argsort()

    percentiles = [0.0] + _get_perc_from_k(k) + [1.0]

    cut_points = np.searchsorted(
        data_perc[data_perc_sort_idx], percentiles, side="right"
    )

    current_cluster = 0
    part = np.zeros(data.shape) - 1

    for i in range(len(cut_points) - 1):
        lim1 = cut_points[i]
        lim2 = cut_points[i + 1]

        part[data_perc_sort_idx[lim1:lim2]] = current_cluster
        current_cluster += 1

    return part.astype(np.uint8)


@njit(cache=True)
def _get_range_n_clusters(
    n_features: int, internal_n_clusters: Iterable[int] = None
) -> np.ndarray:
    """
    Given the number of features it returns a tuple of k values to cluster those
    features into. By default, it generates a tuple of k values from 2 to
    int(np.round(np.sqrt(n_features))) (inclusive). For example, for 25 features,
    it will generate this tuple: (2, 3, 4, 5).

    Args:
        n_features: a positive number representing the number of features that
            will be clustered into different groups/clusters.
        internal_n_clusters: it allows to force a different list of clusters. It
            must be a list of integers. Repeated or invalid values will be dropped,
            such as values lesser than 2 (a singleton partition is not allowed).

    Returns:
        A numpy array with integer values representing numbers of clusters.
    """

    # the one in the list is needed for numba to infer the type
    clusters_range_list = List([1])

    if internal_n_clusters is not None:
        clusters_range_list = List()
        for x in internal_n_clusters:
            clusters_range_list.append(x)

    # keep values larger than one only and remove repeated
    clusters_range_list = list(set([int(x) for x in clusters_range_list if x > 1]))

    # default behavior if no internal_n_clusters is given: return range from
    # 2 to sqrt(n_features)
    if len(clusters_range_list) == 0:
        n_sqrt = int(np.round(np.sqrt(n_features)))
        n_sqrt = min((n_sqrt, 10))
        clusters_range_list = list(range(2, n_sqrt + 1))

    return np.array(clusters_range_list)


@njit(cache=True)
def _get_parts(data: np.ndarray, range_n_clusters: tuple[int]) -> np.ndarray:
    """
    Given a 1d data array, it computes a partition for each k value in the given
    range of clusters. This function only supports numerical data, and it
    always runs run_run_quantile_clustering with the different k values.

    Args:
        data: a 1d data vector. It is assumed that there are no nans.
        range_n_clusters: a tuple with the number of clusters.

    Returns:
        A numpy array with partitions of data, with length equal to the number
        of k values given.
    """
    partitions = []

    for k in range_n_clusters:
        # it doesn't make sense to put each object in its own singleton cluster
        # or create more clusters than number of objects
        if len(data) <= k:
            continue

        part = run_quantile_clustering(data, k)

        # we do not include singleton partitions (only one cluster)
        if len(np.unique(part)) == 1:
            continue

        partitions.append(list(part))

    # This is a hack to get numba compile this function
    if len(partitions) == 0:
        tmp = np.array([[1, 2], [2, 3]], dtype=np.uint8)
        return tmp[np.array([False, False])]

    return np.array(partitions, dtype=np.uint8)


@njit(cache=True)
def cdist_parts(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It implements the same functionality in scipy.spatial.distance.cdist but
    for clustering partitions, and instead of a distance it returns the adjusted
    Rand index (ARI). In other words, it mimics this function call:

        cdist(x, y, metric=ari)

    Args:
        x: a 2d array with m_x clustering partitions in rows and n objects in columns.
        y: a 2d array with m_y clustering partitions in rows and n objects in columns.

    Returns:
        A 2d array with m_x rows and m_y columns and the ARI between each partition pair.
        Each ij entry is equal to ari(x[i], y[j]) for each i and j.
    """
    res = np.zeros((x.shape[0], y.shape[0]))

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = ari(x[i], y[j])

    return res


@njit(cache=True)
def get_coords_from_index(n_obj: int, idx: int) -> tuple[int]:
    """
    Given the number of objects and and index, it returns the row/column position
    of the pairwise matrix. For example, if there are n_obj objects (such as genes),
    a condensed 1d array can be created with pairwise comparisons between genes,
    as well as a squared symmetric matrix. This functions receives the number of objects
    and the index of the condensed array, and returns the coordiates of the squared symmetric
    matrix.

    Args:
        n_obj: the number of objects.
        idx: the index of the condensed pairwise array across all n_obj objects.

    Returns
        A tuple (i, j) with the coordinates of the squared symmetric matrix
        equivalent to the condensed array.
    """
    b = 1 - 2 * n_obj
    x = np.floor((-b - np.sqrt(b ** 2 - 8 * idx)) / 2)
    y = idx + x * (b + x + 2) / 2 + 1
    return int(x), int(y)


@njit(cache=True)
def unravel_index_2d(flat_index: int, shape: tuple[int]) -> tuple[int]:
    if len(shape) != 2:
        raise ValueError("shape has to be of length 2")

    if flat_index >= np.array(shape).prod():
        raise ValueError("index is out of bounds for array with size")

    res = []

    for size in shape[::-1]:
        res.append(flat_index % size)
        flat_index = flat_index // size

    return tuple((res[1], res[0]))


@njit(cache=True, parallel=True)
def _cm(
    x: np.ndarray, y: np.ndarray = None, internal_n_clusters: Iterable[int] = None
) -> np.ndarray:
    """
    This is the main function that computes the Clustermatch coefficient between
    two arrays. This implementation only supports numerical data for
    optimization purposes, but the original implementation can also work with
    categorical data (https://github.com/sinc-lab/clustermatch).

    Args:
        x: an 1d or 2d numerical array with the data. NaN are not supported.
          If it is 2d, then the coefficient is computed for each pair of rows.
        y: an optional 1d numerical array. If x is 1d and y is given, it computes
          the coefficient between x and y.
        internal_n_clusters: a list of integer values indicating the number of
          clusters used to split x and y.

    Returns:
        TODO: UPDATE

        A 1d condensed array of pairwise coefficients. It has size (n * (n - 1))
        / 2, where n is the number of columns in x and y (for example, the
        number of samples for genes).
    """
    if x.ndim == 1 and y is not None:
        assert x.shape == y.shape
        X = np.zeros((2, x.shape[0]))
        X[0, :] = x
        X[1, :] = y
    elif x.ndim == 2:
        X = x
    else:
        raise ValueError("Wrong combination of parameters x and y")

    # get matrix of partitions for each object pair
    parts = []

    for row in X:
        range_n_clusters = _get_range_n_clusters(row.shape[0], internal_n_clusters)
        row_parts = _get_parts(row, range_n_clusters)

        parts.append(row_parts)
        # parts.append([list(x) for x in row_parts])

    # FIXME: ideally, it would be better to also return a ndarray of partitions,
    #  but numba fails to compile with the line below
    # parts = np.array(parts, dtype=np.uint8)

    n = X.shape[0]
    out_size = (n * (n - 1)) // 2
    cm_values = np.empty(out_size)
    cm_values[:] = np.nan

    max_parts = np.zeros((out_size, 2), dtype=np.uint64)
    # max_parts[:] = np.nan

    for idx in prange(cm_values.shape[0]):
        i, j = get_coords_from_index(n, idx)

        # get partitions for the pair of objects
        obji_parts, objj_parts = parts[i], parts[j]

        if obji_parts.shape[0] == 0 or objj_parts.shape[0] == 0:
            max_ari = np.nan
        else:
            comp_values = cdist_parts(obji_parts, objj_parts)
            max_flat_idx = comp_values.argmax()
            max_idx = unravel_index_2d(max_flat_idx, comp_values.shape)

            max_ari = comp_values[max_idx]
            max_parts[idx, :] = max_idx

        cm_values[idx] = max_ari

    return cm_values, max_parts, parts


def to_numpy(x):
    """
    TODO: update
    """
    if x is None:
        return x

    func = getattr(x, "to_numpy", None)
    if not callable(func):
        return x

    return x.to_numpy()


def cm(
    x: np.ndarray,
    y: np.ndarray = None,
    internal_n_clusters: Iterable[int] = None,
    return_parts: bool = False,
):
    """
    This function is a wrapper over _cm, a not-jitted (numba) function that can
    return different value types according to the input given (this is a problem
    with numba).

    Args:
        x: same as in _cm function.
        y: same as in _cm function.
        internal_n_clusters: same as in _cm function.
        return_parts: TODO finish

    Returns:
        TODO: UPDATE

        If x is 2d, then a np.ndarray of size n x n is returned with the
        coefficient value, where n is the number of rows in x. If only a single
        coefficient was computed (for example, x and y were given), then a
        single scalar is returned.
    """

    # convert list to numba.types.List, since reflection is deprecated:
    # https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
    n_clusters = None

    x = to_numpy(x)
    y = to_numpy(y)

    if internal_n_clusters is not None:
        n_clusters = List()
        for k in internal_n_clusters:
            n_clusters.append(k)

    # run optimized _cm function
    cm_values, max_parts, parts = _cm(x, y, n_clusters)

    # return an array of values or a single scalar
    if cm_values.shape[0] == 1:
        if return_parts:
            return cm_values[0], max_parts[0], parts
        else:
            return cm_values[0]

    if return_parts:
        return cm_values, max_parts, parts
    else:
        return cm_values
