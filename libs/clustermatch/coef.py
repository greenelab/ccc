import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ari


def _get_perc_from_k(k):
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


def _get_range_n_clusters(n_features: int, **kwargs) -> list[int]:
    """
    Given the number of features it returns a list of k values to cluster those
    features into. By default, it generates a list of k values from 2 to
    int(np.round(np.sqrt(n_features))) (inclusive). For example, for 25 features,
    it will generate this list: [2, 3, 4, 5].

    Args:
        n_features: a positive number representing the number of features that
            will be clustered into different groups/clusters.
        internal_n_clusters: it allows to force a different list of clusters. It
            can be a list/tuple of integers (floats will be converted into int) or a
            range object, or even an integer (in that case, it will return a list
            with that integer). Invalid or repeated values will be dropped, such
            as values lesser than 2 (a singleton partition is not allowed)
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
        int(x) for x in clusters_range_list
        if x > 1 and not (x in seen or seen_add(x))
    ]

    if len(clusters_range_list) == 0:
        n_sqrt = int(np.round(np.sqrt(n_features)))
        clusters_range_list = range(2, n_sqrt + 1)

    return tuple(clusters_range_list)


def _get_internal_parts(data_obj, range_n_clusters, **kwargs):
    partitions = []

    for k in range_n_clusters:
        # TODO: the commented out code below, I think, it's useful for
        #  pd.Series/DataFrames, not np.arrays
        # if len(data_obj) <= k:
        #     part = np.array([np.nan] * len(data_obj))
        # else:
        part = run_quantile_clustering(data_obj, k, **kwargs)

        partitions.append(part)

    return np.array(partitions)


def _compute_ari(part1, part2):
    if np.isnan(part1).any() or len(part1) == 0:
        return 0.0

    return ari(part1, part2)


def cm(obj1, obj2, **kwargs):
    range_n_clusters = _get_range_n_clusters(len(obj1), **kwargs)

    obj1_parts = _get_internal_parts(obj1, range_n_clusters, **kwargs)
    obj2_parts = _get_internal_parts(obj2, range_n_clusters, **kwargs)

    comp_values = cdist(obj1_parts, obj2_parts, metric=_compute_ari)

    max_pos = np.unravel_index(comp_values.argmax(), comp_values.shape)
    max_ari = comp_values[max_pos]

    # if the partition that maximizes the ARI in either of the two input vectors
    # has only one cluster (for example, all the values are the same), then
    obj1_max_part = obj1_parts[max_pos[0]]
    obj2_max_part = obj2_parts[max_pos[1]]

    if len(np.unique(obj1_max_part)) == 1 or len(np.unique(obj2_max_part)) == 1:
        return 0.0

    return max_ari


# def _calculate_sub_simmatrix(data, idx_range, sim_func='cm', return_pvalue=False, min_n_common_features=3, **kwargs):
#     p_dist = []
#     p_dist_pvalue = []
#     n_objects = data.shape[0]
#
#     if sim_func == 'cm':
#         similarity_func = cm
#     elif sim_func == 'pearson':
#         similarity_func = get_pearson
#     elif sim_func == 'shared_objects':
#         similarity_func = get_shared_objects
#     else:
#         raise ValueError('Invalid sim_func')
#
#     for idx in idx_range:
#         obj1_idx, obj2_idx = row_col_from_condensed_index(n_objects, idx)
#
#         obj1 = data[obj1_idx]
#         obj2 = data[obj2_idx]
#
#         common_features, n_common_features = _get_common_features(obj1, obj2)
#
#         if n_common_features < min_n_common_features:
#             sim_values = (0.0, 1.0) # sim value and pvalue
#         else:
#             sim_values = similarity_func(obj1[common_features], obj2[common_features], **kwargs)
#
#         p_dist.append(sim_values[0])
#
#         if return_pvalue:
#             p_dist_pvalue.append(sim_values[1])
#
#     return p_dist, p_dist_pvalue


# def calculate_simmatrix(data, fill_diag_value=1.0, n_jobs=1, **kwargs):
#     data_index = None
#     if hasattr(data, 'index'):
#         data_index = data.index.tolist()
#
#     if hasattr(data, 'values'):
#         data = data.values
#
#     # FIXME: quantiles clustering is only for 1d comparisons. Use kmeans for n-dimensional.
#     # kwargs['clustering_method'] = _get_clustering_method(**kwargs)
#
#     n_objects = data.shape[0]
#
#     p_dist_len = int((n_objects * (n_objects - 1)) / 2)
#
#     # FIXME: set n_jobs according to data size. Do some performance test with unit tests
#     n_cpus = n_jobs if n_jobs > 0 else cpu_count()
#
#     step = int(np.ceil(p_dist_len / n_cpus))
#     p_dist_range = range(0, p_dist_len, step)
#
#     p_dist_values = Parallel(n_jobs=n_jobs)(
#         delayed(_calculate_sub_simmatrix)(data, idx_range, **kwargs)
#         for idx_range in [range(s, min(s + p_dist_range.step, p_dist_range.stop)) for s in p_dist_range]
#     )
#
#     p_dist = []
#     p_dist_pval = []
#     for p, pval in p_dist_values:
#         p_dist.extend(p)
#
#         if len(pval) > 0:
#             p_dist_pval.extend(pval)
#
#     p_dist = np.array(p_dist)
#     p_dist_pval = np.array(p_dist_pval)
#
#     return_pvalue = kwargs.get('return_pvalue', False)
#
#     if data_index is not None:
#         sqmatrix = get_squareform(p_dist, fill_diag_value)
#         sim_matrix = pd.DataFrame(
#             sqmatrix,
#             index=data_index,
#             columns=data_index
#         )
#
#         if return_pvalue:
#             sqmatrix_pval = get_squareform(p_dist_pval, np.nan)
#             pval_matrix = pd.DataFrame(
#                 sqmatrix_pval,
#                 index=data_index,
#                 columns=data_index
#             )
#     else:
#         sim_matrix = p_dist
#         pval_matrix = p_dist_pval
#
#     if return_pvalue:
#         return sim_matrix, pval_matrix
#     else:
#         return sim_matrix
