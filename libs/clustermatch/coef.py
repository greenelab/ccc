"""
Contains function that implement the Clustermatch coefficient
(https://doi.org/10.1093/bioinformatics/bty899).
"""
import numpy as np
from numba import njit, types
from scipy.spatial.distance import cdist


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

    return part


@njit(cache=True)
def _get_range_n_clusters(n_features: int, internal_n_clusters: list = None) -> tuple[int]:
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
    clusters_range_list = [1]

    if internal_n_clusters is not None:
        clusters_range_list = internal_n_clusters

    # keep values larger than one only and remove repeated
    clusters_range_list = list(set([
        int(x) for x in clusters_range_list if x > 1
    ]))

    # default behavior if no internal_n_clusters is given: return range from
    # 2 to sqrt(n_features)
    if len(clusters_range_list) == 0:
        n_sqrt = int(np.round(np.sqrt(n_features)))
        n_sqrt = min((n_sqrt, 10))
        clusters_range_list = list(range(2, n_sqrt + 1))

    return np.array(clusters_range_list, dtype="uint")


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
        # TODO: the commented out code below, I think, it's useful for
        #  pd.Series/DataFrames, not np.arrays
        # it doesn't make sense to put each object in its own singleton cluster
        # or create more clusters than number of objects
        if len(data) <= k:
            continue

        part = run_quantile_clustering(data, k)

        # we do not include singleton partitions (only one cluster)
        if len(np.unique(part)) == 1:
            continue

        partitions.append(part)

    # TODO: use np.int8 or something like that as dtype
    return np.array(partitions)


def _isempty(row):
    return np.array([x is None or (np.isreal(x) and np.isnan(x)) for x in row])


def _get_common_features(obj1, obj2):
    obj1_notnan = np.logical_not(_isempty(obj1))
    obj2_notnan = np.logical_not(_isempty(obj2))

    common_features = np.logical_and(obj1_notnan, obj2_notnan)
    n_common_features = common_features.sum()

    return common_features, n_common_features


def cm(x, y=None, precompute_parts=False, **kwargs):
    """
    This is the main function that computes the Clustermatch coefficient between
    two arrays. This implementation only supports numerical data for
    optimization purposes, but it can also work with categorical data in the
    original implementation (https://github.com/sinc-lab/clustermatch).

    Args:
        x:
        y:
        precompute_parts: this parameter should be set to True only in the case
            where there are no missing data in the input matrix. Otherwise, it
            will generate different results than running it this parameter set
            to False.

    TODO: finish
    """
    if x.ndim == 1 and y is not None:
        assert x.shape == y.shape
        x = np.array([x, y])

    # TODO: (future) if x is matrix and y a vector, then
    #  we can do all rows in x against the vector in y?

    # get matrix of partitions for each object pair
    if precompute_parts:
        parts = []

        for row in x:
            range_n_clusters = _get_range_n_clusters(row.shape[0], **kwargs)
            row_parts = _get_parts(row, range_n_clusters)
            parts.append(row_parts)

    # TODO: split parts with chunker for the support of multiple cores

    # parts is a dictionary?
    # key: (obj_i, obj_j)
    # value: (shared_idx, obj_i_parts, obj_j_parts)
    #
    # OR
    # parts is a list of matrices (several partitions per object): np.array(...)
    #  !!! none of the parts should contain only one cluster

    n = x.shape[0]
    out_size = (n * (n - 1)) // 2
    cm_values = np.empty(out_size)
    cm_values[:] = np.nan

    idx = 0
    for i in range(x.shape[0] - 1):
        for j in range(i + 1, x.shape[0]):
            # get partitions for the pair of objects
            if precompute_parts:
                obji_parts, objj_parts = parts[i], parts[j]
            else:
                obji, objj = x[i], x[j]

                common_features, n_common_features = _get_common_features(obji, objj)

                obji = obji[common_features]
                objj = objj[common_features]

                range_n_clusters = _get_range_n_clusters(n_common_features, **kwargs)
                obji_parts = _get_parts(obji, range_n_clusters)
                objj_parts = _get_parts(objj, range_n_clusters)

            if obji_parts.shape[0] == 0 or objj_parts.shape[0] == 0:
                max_ari = np.nan
            else:

                comp_values = cdist(obji_parts, objj_parts, metric=ari)

                max_pos = np.unravel_index(comp_values.argmax(), comp_values.shape)
                max_ari = comp_values[max_pos]

                # TODO: use this to return stats
                # get the partition in obj1 and the partition in obj2 that maximized ari
                # obj1_max_part = obji_parts[max_pos[0]]
                # obj2_max_part = objj_parts[max_pos[1]]

            cm_values[idx] = max_ari
            idx += 1

    if cm_values.shape[0] == 1:
        return cm_values[0]

    return cm_values

    # common_features, n_common_features = _get_common_features(obj1, obj2)
    #
    # obj1 = obj1[common_features]
    # obj2 = obj2[common_features]
    # range_n_clusters = _get_range_n_clusters(n_common_features, **kwargs)
    #
    # obj1_parts = _get_internal_parts(obj1, range_n_clusters, **kwargs)
    # obj2_parts = _get_internal_parts(obj2, range_n_clusters, **kwargs)
    #
    # comp_values = cdist(obj1_parts, obj2_parts, metric=_compute_ari)
    #
    # max_pos = np.unravel_index(comp_values.argmax(), comp_values.shape)
    # max_ari = comp_values[max_pos]
    #
    # # get the partition in obj1 and the partition in obj2 that maximized ari
    # obj1_max_part = obj1_parts[max_pos[0]]
    # obj2_max_part = obj2_parts[max_pos[1]]
    #
    # # if the partition that maximizes the ARI in either of the two input vectors
    # # has only one cluster (for example, all the values are the same), then the
    # # coefficient is zero
    # if len(np.unique(obj1_max_part)) == 1 or len(np.unique(obj2_max_part)) == 1:
    #     return 0.0
    #
    # return max_ari
