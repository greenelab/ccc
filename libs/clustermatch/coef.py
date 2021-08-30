import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ari


def _get_perc_from_k(k):
    return [(1.0 / k) * i for i in range(1, k)]


def run_quantile_clustering(data, k, **kwargs):
    """
    TODO

    it can return less clusters than specified if all the values are the same
    """
    data_perc = stats.rankdata(data, "average") / len(data)
    data_perc_sort_idx = np.argsort(data_perc)

    # data_perc = data
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


# def _isempty(row):
#     return np.array([x is None or (np.isreal(x) and np.isnan(x)) for x in row])


# def _get_common_features(obj1, obj2):
#     obj1_notnan = np.logical_not(_isempty(obj1))
#     obj2_notnan = np.logical_not(_isempty(obj2))
#
#     common_features = np.logical_and(obj1_notnan, obj2_notnan)
#     n_common_features = common_features.sum()
#
#     return common_features, n_common_features


def _get_range_n_clusters(n_common_features, **kwargs):
    internal_n_clusters = kwargs.get("internal_n_clusters")

    if internal_n_clusters is None:
        estimated_k = int(np.floor(np.sqrt(n_common_features)))
        estimated_k = np.min((estimated_k, 10))
        range_n_clusters = range(2, np.max((estimated_k, 3)))
    elif isinstance(internal_n_clusters, (tuple, list, range)):
        # TODO: test the case where this is a range, because it's a generator
        range_n_clusters = internal_n_clusters
    elif isinstance(internal_n_clusters, int):
        # TODO: test this case
        range_n_clusters = (internal_n_clusters,)
    else:
        raise ValueError("n_clusters is invalid")

    return range_n_clusters


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
