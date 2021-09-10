"""
Contains function that implement the Clustermatch coefficient
(https://doi.org/10.1093/bioinformatics/bty899).
"""
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
def rank(data):
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
    n_features: int, internal_n_clusters: list = None
) -> tuple[int]:
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
def cdist_parts(x, y):
    res = np.zeros((x.shape[0], y.shape[0]))

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = ari(x[i], y[j])

    return res


@njit(cache=True)
def get_coords_from_index(n_obj, idx):
    """
    TODO: finish
    """
    b = 1 -2*n_obj
    x = np.floor((-b - np.sqrt(b**2 - 8*idx))/2)
    y = idx + x*(b + x + 2)/2 + 1
    return int(x), int(y)


@njit(cache=True, parallel=True)
def _cm(x, y=None, internal_n_clusters: list = None):
    """
    This is the main function that computes the Clustermatch coefficient between
    two arrays. This implementation only supports numerical data for
    optimization purposes, but it can also work with categorical data in the
    original implementation (https://github.com/sinc-lab/clustermatch).

    Args:
        x:
        y:
        internal_n_clusters:

    TODO: finish
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

    # TODO: (future) if x is matrix and y a vector, then
    #  we can do all rows in x against the vector in y?

    # get matrix of partitions for each object pair
    parts = []

    for row in X:
        range_n_clusters = _get_range_n_clusters(row.shape[0], internal_n_clusters)
        row_parts = _get_parts(row, range_n_clusters)

        parts.append(row_parts)

    n = X.shape[0]
    out_size = (n * (n - 1)) // 2
    cm_values = np.empty(out_size)
    cm_values[:] = np.nan

    for idx in prange(cm_values.shape[0]):
        i, j = get_coords_from_index(n, idx)

        # get partitions for the pair of objects
        obji_parts, objj_parts = parts[i], parts[j]

        if obji_parts.shape[0] == 0 or objj_parts.shape[0] == 0:
            max_ari = np.nan
        else:
            comp_values = cdist_parts(obji_parts, objj_parts)

            # max_pos = np.unravel_index(comp_values.argmax(), comp_values.shape)
            max_ari = np.amax(comp_values)
            # max_pos = np.where(comp_values == max_ari)
            # max_ari = comp_values[max_pos]

            # TODO: use this to return stats
            # get the partition in obj1 and the partition in obj2 that maximized ari
            # obj1_max_part = obji_parts[max_pos[0]]
            # obj2_max_part = objj_parts[max_pos[1]]

        cm_values[idx] = max_ari

    return cm_values


def cm(x, y=None, internal_n_clusters: list = None):
    """
    This function is a not-jitted function that can return different value types
    according to the input given.

    TODO: finish
    """

    # convert list to numba.types.List, since reflection is deprecated:
    # https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
    n_clusters = None

    if internal_n_clusters is not None:
        n_clusters = List()
        for k in internal_n_clusters:
            n_clusters.append(k)

    # run optimized _cm function
    cm_values = _cm(x, y, n_clusters)

    # return an array of values or a single scalar, which depends on the input
    # data shape
    if cm_values.shape[0] == 1:
        return cm_values[0]

    return cm_values
