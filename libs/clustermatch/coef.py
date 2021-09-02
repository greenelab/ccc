"""
Contains function that implement the Clustermatch coefficient
(https://doi.org/10.1093/bioinformatics/bty899).
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ari


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
    data_perc = stats.rankdata(data, "average") / len(data)
    data_perc_sort_idx = np.argsort(data_perc)

    percentiles = [0.0] + _get_perc_from_k(k) + [1.0]

    cut_points = np.searchsorted(
        data_perc[data_perc_sort_idx], percentiles, side="right"
    )

    current_cluster = 0
    part = np.zeros(data.shape, dtype=float) - 1

    for i in range(len(cut_points) - 1):
        lim1 = cut_points[i]
        lim2 = cut_points[i + 1]

        part[data_perc_sort_idx[lim1:lim2]] = current_cluster
        current_cluster += 1

    return part


def _get_range_n_clusters(n_features: int, **kwargs) -> tuple[int]:
    """
    Given the number of features it returns a tuple of k values to cluster those
    features into. By default, it generates a tuple of k values from 2 to
    int(np.round(np.sqrt(n_features))) (inclusive). For example, for 25 features,
    it will generate this tuple: (2, 3, 4, 5).

    Args:
        n_features: a positive number representing the number of features that
            will be clustered into different groups/clusters.
        internal_n_clusters: it allows to force a different list of clusters. It
            can be a list/tuple of integers (floats will be converted into int) or a
            range object, or even an integer (in that case, it will return a list
            with that integer). Invalid or repeated values will be dropped, such
            as values lesser than 2 (a singleton partition is not allowed).

    Returns:
        A tuple with integer values representing number of clusters.
    """
    internal_n_clusters = kwargs.get("internal_n_clusters")

    if isinstance(internal_n_clusters, (tuple, list, range)):
        clusters_range_list = internal_n_clusters
    elif isinstance(internal_n_clusters, int):
        clusters_range_list = [internal_n_clusters]
    else:
        clusters_range_list = []

    # keep values larger than one only and remove repeated
    # code to remove repeated values taken from: https://stackoverflow.com/a/480227
    seen = set()
    seen_add = seen.add

    clusters_range_list = [
        int(x) for x in clusters_range_list if x > 1 and not (x in seen or seen_add(x))
    ]

    if len(clusters_range_list) == 0:
        n_sqrt = int(np.round(np.sqrt(n_features)))
        clusters_range_list = range(2, n_sqrt + 1)

    return tuple(clusters_range_list)


def _get_internal_parts(data: np.ndarray, range_n_clusters: tuple[int]) -> np.ndarray:
    """
    Given a 1d data array, it computes a partition for each k value in the given
    range of clusters. This function only supports numerical data, and it
    always runs run_run_quantile_clustering with the different k values.

    Args:
        data: a 1d data vector.
        range_n_clusters: a tuple with the number of clusters.

    Returns:
        A numpy array with partitions of data, with length equal to the number
        of k values given.
    """
    partitions = []

    for k in range_n_clusters:
        # TODO: the commented out code below, I think, it's useful for
        #  pd.Series/DataFrames, not np.arrays
        # if len(data_obj) <= k:
        #     part = np.array([np.nan] * len(data_obj))
        # else:
        part = run_quantile_clustering(data, k)

        partitions.append(part)

    # TODO: use np.int8 or something like that as dtype
    return np.array(partitions)


def _compute_ari(part1, part2):
    # TODO: not sure why I have this here, test it!
    # if np.isnan(part1).any() or len(part1) == 0:
    #     return 0.0

    # TODO: maybe replace with my own ari implementation, which also fixes some issues.
    #  this will also be necessary for numba.
    return ari(part1, part2)


def cm(obj1, obj2, **kwargs):
    """
    This is the main function that computes the Clustermatch coefficient between
    two arrays. This implementation only supports numerical data for
    optimization purposes, but it can also work with categorical data in the
    original implementation (https://github.com/sinc-lab/clustermatch).

    TODO: this function might check whether obj1 (which should probably by x, and obj2 be y)
     is one or two dimensional, and if y is given; if 2d, use optimized approach to compute
     cm on all column pairs

    TODO: finish
    """
    range_n_clusters = _get_range_n_clusters(len(obj1), **kwargs)

    obj1_parts = _get_internal_parts(obj1, range_n_clusters, **kwargs)
    obj2_parts = _get_internal_parts(obj2, range_n_clusters, **kwargs)

    comp_values = cdist(obj1_parts, obj2_parts, metric=_compute_ari)

    max_pos = np.unravel_index(comp_values.argmax(), comp_values.shape)
    max_ari = comp_values[max_pos]

    # get the partition in obj1 and the partition in obj2 that maximized ari
    obj1_max_part = obj1_parts[max_pos[0]]
    obj2_max_part = obj2_parts[max_pos[1]]

    # if the partition that maximizes the ARI in either of the two input vectors
    # has only one cluster (for example, all the values are the same), then the
    # coefficient is zero
    if len(np.unique(obj1_max_part)) == 1 or len(np.unique(obj2_max_part)) == 1:
        return 0.0

    return max_ari
