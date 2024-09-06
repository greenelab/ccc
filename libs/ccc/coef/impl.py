"""
Contains function that implement the Clustermatch Correlation Coefficient (CCC).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import Iterable, Union

import numpy as np
from numpy.typing import NDArray
from numba import njit
from numba.typed import List

from ccc.pytorch.core import unravel_index_2d
# from ccc.sklearn.metrics import adjusted_rand_index as ari
from ccc.sklearn.metrics_gpu import adjusted_rand_index as ari
from ccc.scipy.stats import rank
from ccc.utils import chunker, DummyExecutor


@njit(cache=True, nogil=True)
def get_perc_from_k(k: int) -> list[float]:
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


# @njit(cache=True, nogil=True)
def run_quantile_clustering(data: NDArray, k: int) -> NDArray[np.int16]:
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
    data_sorted = np.argsort(data, kind="quicksort")
    data_rank = rank(data, data_sorted)
    data_perc = data_rank / len(data)

    # percentiles = [0.0] + get_perc_from_k(k) + [1.0]
    percentiles = get_perc_from_k(k)
    # print(f"CPU percentages: {str(percentiles)}")

    # cut_points = np.searchsorted(data_perc[data_sorted], percentiles, side="right")
    #
    # current_cluster = 0
    # part = np.zeros(data.shape, dtype=np.int16) - 1
    #
    # for i in range(len(cut_points) - 1):
    #     lim1 = cut_points[i]
    #     lim2 = cut_points[i + 1]
    #
    #     part[data_sorted[lim1:lim2]] = current_cluster
    #     current_cluster += 1
    bins = np.quantile(data, percentiles)
    part = np.digitize(data, bins, right=True)
    return part


# @njit(cache=True, nogil=True)
def get_range_n_clusters(
    n_features: int, internal_n_clusters: Iterable[int] = None
) -> NDArray[np.uint8]:
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

    if internal_n_clusters is not None:
        # remove k values that are invalid
        clusters_range_list = list(
            set([int(x) for x in internal_n_clusters if 1 < x < n_features])
        )
    else:
        # default behavior if no internal_n_clusters is given: return range from
        # 2 to sqrt(n_features)
        n_sqrt = int(np.round(np.sqrt(n_features)))
        n_sqrt = min((n_sqrt, 10))
        clusters_range_list = list(range(2, n_sqrt + 1))

    return np.array(clusters_range_list, dtype=np.uint16)


# @njit(cache=True, nogil=True)
def get_parts(
    data: NDArray, range_n_clusters: tuple[int], data_is_numerical: bool = True
) -> NDArray[np.int16]:
    """
    Given a 1d data array, it computes a partition for each k value in the given
    range of clusters. If partitions with only one cluster are returned (singletons),
    then the returned array will have negative values.

    Args:
        data: a 1d data vector. It is assumed that there are no nans.
        range_n_clusters: a tuple with the number of clusters.
        data_is_numerical: indicates whether data is numerical (True) or categorical (False)

    Returns:
        A numpy array with shape (number of clusters, data rows) with
        partitions of data.

        Partitions could have negative values in some scenarios, with different
        meanings: -1 is used for categorical data, where only one partition is generated
        and the rest (-1) are marked as "empty". -2 is used when singletons have been
        detected (partitions with one cluster), usually because of problems with the
        input data (it has all the same values, for example).
    """
    parts = np.zeros((len(range_n_clusters), data.shape[0]), dtype=np.int16) - 1

    if data_is_numerical:
        for idx in range(len(range_n_clusters)):
            k = range_n_clusters[idx]
            parts[idx] = run_quantile_clustering(data, k)

        # remove singletons by putting a -2 as values
        partitions_ks = np.array([len(np.unique(p)) for p in parts])
        parts[partitions_ks == 1, :] = -2
    else:
        # if the data is categorical, then the encoded feature is already the partition
        # only the first partition is filled, the rest will be -1 (missing)
        parts[0] = data.astype(np.int16)

    return parts


def get_feature_parts(params):
    """
    Given a list of parameters, it returns the partitions for each feature. The goal
    of this function is to parallelize the partitioning step (get_parts function).

    Args:
        params: a list of tuples with three elements: 1) a tuple with the feature
            index, the cluster index and the number of clusters (k), 2) the data for the
            feature, and 3) a boolean indicating whether the feature is numerical or not.

    Returns:
        A 2d array with the partitions (rows) for the selected features and number of
        clusters.
    """
    n_objects = params[0][1].shape[0]
    parts = np.zeros((len(params), n_objects), dtype=np.int16) - 1

    # iterate over a list of tuples that indicate a feature-k pair
    for p_idx, p in enumerate(params):
        # the first element is a tuple with the feature index, the cluster index and the
        # number of clusters (k)
        info = p[0]
        # f_idx = info[0]
        c_idx = info[1]
        c = info[2]
        range_n_clusters = np.array([c], dtype=np.uint16)

        # the second element is the data for the feature
        data = p[1]

        # the third element is a boolean indicating whether the feature is numerical
        numerical_data_type = p[2]

        # if the feature is categorical, then only the first partition is filled
        if not numerical_data_type and c_idx > 0:
            continue

        parts[p_idx] = get_parts(data, range_n_clusters, numerical_data_type)

    return parts


def cdist_parts_basic(x: NDArray, y: NDArray) -> NDArray[float]:
    """
    It implements the same functionality in scipy.spatial.distance.cdist but
    for clustering partitions, and instead of a distance it returns the adjusted
    Rand index (ARI). In other words, it mimics this function call:

        cdist(x, y, metric=ari)

    Only partitions with positive labels (> 0) are compared. This means that
    partitions marked as "singleton" or "empty" (categorical data) are not
    compared. This has the effect of leaving an ARI of 0.0 (zero).

    Args:
        x: a 2d array with m_x clustering partitions in rows and n objects in
          columns.
        y: a 2d array with m_y clustering partitions in rows and n objects in
          columns.

    Returns:
        A 2d array with m_x rows and m_y columns and the ARI between each
        partition pair. Each ij entry is equal to ari(x[i], y[j]) for each i
        and j.
    """
    res = np.zeros((x.shape[0], y.shape[0]))

    for i in range(res.shape[0]):
        if x[i, 0] < 0:
            continue

        for j in range(res.shape[1]):
            if y[j, 0] < 0:
                continue

            res[i, j] = ari(x[i], y[j])

    return res


def cdist_parts_parallel(
    x: NDArray, y: NDArray, executor: ThreadPoolExecutor
) -> NDArray[float]:
    """
    It parallelizes cdist_parts_basic function.

    Args:
        x: same as in cdist_parts_basic
        y: same as in cdist_parts_basic
        executor: a pool executor where jobs will be submitted.

    Results:
        Same as in cdist_parts_basic.
    """
    res = np.zeros((x.shape[0], y.shape[0]))

    inputs = get_chunks(res.shape[0], executor._max_workers, 1)

    tasks = {executor.submit(cdist_parts_basic, x[idxs], y): idxs for idxs in inputs}
    for t in as_completed(tasks):
        idx = tasks[t]
        res[idx, :] = t.result()

    return res


@njit(cache=True, nogil=True)
def get_coords_from_index(n_obj: int, idx: int) -> tuple[int]:
    """
    Given the number of objects and an index, it returns the row/column
    position of the pairwise matrix. For example, if there are n_obj objects
    (such as genes), a condensed 1d array can be created with pairwise
    comparisons between genes, as well as a squared symmetric matrix. This
    function receives the number of objects and the index of the condensed
    array, and returns the coordiates of the squared symmetric matrix.

    Args:
        n_obj: the number of objects.
        idx: the index of the condensed pairwise array across all n_obj objects.

    Returns
        A tuple (i, j) with the coordinates of the squared symmetric matrix
        equivalent to the condensed array.
    """
    b = 1 - 2 * n_obj
    x = np.floor((-b - np.sqrt(b**2 - 8 * idx)) / 2)
    y = idx + x * (b + x + 2) / 2 + 1
    return int(x), int(y)


def get_chunks(
    iterable: Union[int, Iterable], n_threads: int, ratio: float = 1
) -> Iterable[Iterable[int]]:
    """
    It splits elements in an iterable in chunks according to the number of
    CPU cores available for parallel processing.

    Args:
        iterable: an iterable to be split in chunks. If it is an integer, it
            will split the iterable given by np.arange(iterable).
        n_threads: number of threads available for parallelization.
        ratio: a ratio that allows to increase the number of splits given
            n_threads. For example, with ratio=1, the function will just split
            the iterable in n_threads chunks. If ratio is larger than 1, then
            it will split in n_threads * ratio chunks.

    Results:
        Another iterable with chunks according to the arguments given. For
        example, if iterable is [0, 1, 2, 3, 4, 5] and n_threads is 2, it will
        return [[0, 1, 2], [3, 4, 5]].
    """
    if isinstance(iterable, int):
        iterable = np.arange(iterable)

    n = len(iterable)
    expected_n_chunks = n_threads * ratio

    res = list(chunker(iterable, int(np.ceil(n / expected_n_chunks))))

    while len(res) < expected_n_chunks <= n:
        # look for an element in res that can be split in two
        idx = 0
        while len(res[idx]) == 1:
            idx = idx + 1

        new_chunk = get_chunks(res[idx], 2)
        res[idx] = new_chunk[0]
        res.insert(idx + 1, new_chunk[1])

    return res


def get_feature_type_and_encode(feature_data: NDArray) -> tuple[NDArray, bool]:
    """
    Given the data of one feature as a 1d numpy array (it could also be a pandas.Series),
    it returns the same data if it is numerical (float, signed or unsigned integer) or an
    encoded version if it is categorical (each category value has a unique integer starting from
    zero).

    Args:
        feature_data: a 1d array with data.

    Returns:
        A tuple with two elements:
          1. the feature data: same as input if numerical, encoded version if not numerical.
          2. A boolean indicating whether the feature data is numerical or not.
    """
    data_type_is_numerical = feature_data.dtype.kind in ("f", "i", "u")
    if data_type_is_numerical:
        return feature_data, data_type_is_numerical

    # here np.unique with return_inverse encodes categorical values into numerical ones
    return np.unique(feature_data, return_inverse=True)[1], data_type_is_numerical


def compute_ccc(obj_parts_i: NDArray, obj_parts_j: NDArray, cdist_func):
    """
    Given a set of partitions for two features, it computes the CCC coefficient.

    Args:
        obj_parts_i: a 2d array with partitions for one feature. Each row is a
            partition, and each column is an object.
        obj_parts_j: a 2d array with partitions for another feature. Each row is
            a partition, and each column is an object.
        cdist_func: a function that computes the distance between partitions. It
            can be either cdist_parts_basic or cdist_parts_parallel.

    Returns:
        A tuple with two elements: 1) the CCC coefficient, and 2) the indexes
        of the partitions that maximized the coefficient.
    """
    comp_values = cdist_func(
        obj_parts_i,
        obj_parts_j,
    )
    max_flat_idx = comp_values.argmax()
    max_idx = unravel_index_2d(max_flat_idx, comp_values.shape)

    return max(comp_values[max_idx], 0.0), max_idx


def compute_ccc_perms(params) -> NDArray[float]:
    """
    Similar to compute_ccc (with same parameters), but it computes the CCC coefficient
    by permuting the partitions of one of the features n_perms times.

    Args:
        params: a tuple with four elements: 1) the index of the permutations, 2) the
            partitions of one of the features, 3) the partitions of the other feature,
            and 4) the number of permutations to perform.

    Returns:
        The CCC coefficient values using the permuted partitions of one of the features.
    """
    # since this function can be parallelized across different processes, make sure
    # the random number generator is initialized with a different seed for each process
    rng = np.random.default_rng()

    _, obj_parts_i, obj_parts_j, n_perms = params

    n_objects = obj_parts_i.shape[1]
    ccc_perm_values = np.full(n_perms, np.nan, dtype=float)

    for idx in range(n_perms):
        perm_idx = rng.permutation(n_objects)

        # generate a random permutation of the partitions of one
        # variable/feature
        obj_parts_j_permuted = np.full_like(obj_parts_j, np.nan)
        for it in range(obj_parts_j.shape[0]):
            obj_parts_j_permuted[it] = obj_parts_j[it][perm_idx]

        # compute the CCC using the permuted partitions
        ccc_perm_values[idx] = compute_ccc(
            obj_parts_i, obj_parts_j_permuted, cdist_parts_basic
        )[0]

    return ccc_perm_values


def compute_coef(params):
    """
    Given a list of indexes representing each a pair of
    objects/rows/genes, it computes the CCC coefficient for
    each of them. This function is supposed to be used to parallelize
    processing.

    Args:
        params: a tuple with eight elements: 1) the indexes of the features
            to compare, 2) the number of features, 3) the partitions for each
            feature, 4) the number of permutations to compute the p-value, 5)
            the number of threads to use for parallelization, 6) the ratio
            between the number of chunks and the number of threads, 7) the
            executor to use for cdist parallelization, and 8) the executor to use
            for parallelization of permutations.

    Returns:
        Returns a tuple with three arrays. The first array has the CCC
        coefficients, the second array has the indexes of the partitions that
        maximized the coefficient, and the third array has the p-values.
    """
    (
        idx_list,
        n_features,
        parts,
        pvalue_n_perms,
        default_n_threads,
        n_chunks_threads_ratio,
        cdist_executor,
        executor,
    ) = params

    cdist_func = cdist_parts_basic
    if cdist_executor is not False:

        def cdist_func(x, y):
            return cdist_parts_parallel(x, y, cdist_executor)

    n_idxs = len(idx_list)
    max_ari_list = np.full(n_idxs, np.nan, dtype=float)
    max_part_idx_list = np.zeros((n_idxs, 2), dtype=np.uint64)
    pvalues = np.full(n_idxs, np.nan, dtype=float)

    for idx, data_idx in enumerate(idx_list):
        i, j = get_coords_from_index(n_features, data_idx)

        # get partitions for the pair of objects
        obji_parts, objj_parts = parts[i], parts[j]

        # compute ari only if partitions are not marked as "missing"
        # (negative values), which is assigned when partitions have
        # one cluster (usually when all data in the feature has the same
        # value).
        if obji_parts[0, 0] == -2 or objj_parts[0, 0] == -2:
            continue

        # compare all partitions of one object to the all the partitions
        # of the other object, and get the maximium ARI
        max_ari_list[idx], max_part_idx_list[idx] = compute_ccc(
            obji_parts, objj_parts, cdist_func
        )

        # compute p-value if requested
        if pvalue_n_perms is not None and pvalue_n_perms > 0:
            # with ThreadPoolExecutor(max_workers=pvalue_n_jobs) as executor_perms:
            # select the variable that generated more partitions as the one
            # to permute
            obj_parts_sel_i = obji_parts
            obj_parts_sel_j = objj_parts
            if (obji_parts[:, 0] >= 0).sum() > (objj_parts[:, 0] >= 0).sum():
                obj_parts_sel_i = objj_parts
                obj_parts_sel_j = obji_parts

            p_ccc_values = np.full(pvalue_n_perms, np.nan, dtype=float)
            p_inputs = get_chunks(
                pvalue_n_perms, default_n_threads, n_chunks_threads_ratio
            )
            p_inputs = [
                (
                    i,
                    obj_parts_sel_i,
                    obj_parts_sel_j,
                    len(i),
                )
                for i in p_inputs
            ]

            for params, p_ccc_val in zip(
                p_inputs,
                executor.map(
                    compute_ccc_perms,
                    p_inputs,
                ),
            ):
                p_idx = params[0]

                p_ccc_values[p_idx] = p_ccc_val

            # compute p-value
            pvalues[idx] = (np.sum(p_ccc_values >= max_ari_list[idx]) + 1) / (
                pvalue_n_perms + 1
            )

    return max_ari_list, max_part_idx_list, pvalues


def get_n_workers(n_jobs: int | None) -> int:
    """
    Helper function to get the number of workers for parallel processing.

    Args:
        n_jobs: value specified by the main ccc function.
    Returns:
        The number of workers to use for parallel processing
    """
    n_cpu_cores = os.cpu_count()
    if n_cpu_cores is None:
        raise ValueError("Could not determine the number of CPU cores. Please specify a positive value of n_jobs")

    n_workers = n_cpu_cores
    if n_jobs is None:
        return n_workers

    n_workers = os.cpu_count() + n_jobs if n_jobs < 0 else n_jobs

    if n_workers < 1:
        raise ValueError(f"The number of threads/processes to use must be greater than 0. Got {n_workers}."
                         "Please check the n_jobs argument provided")

    return n_workers


def ccc(
    x: NDArray,
    y: NDArray = None,
    internal_n_clusters: Union[int, Iterable[int]] = None,
    return_parts: bool = False,
    n_chunks_threads_ratio: int = 1,
    n_jobs: int = 1,
    pvalue_n_perms: int = None,
    partitioning_executor: str = "thread",
) -> tuple[NDArray[float], NDArray[float], NDArray[np.uint64], NDArray[np.int16]]:
    """
    This is the main function that computes the Clustermatch Correlation
    Coefficient (CCC) between two arrays. The implementation supports numerical
    and categorical data.

    Args:
        x: 1d or 2d numerical array with the data. NaN are not supported.
          If it is 2d, then the coefficient is computed for each pair of rows
          (in case x is a numpy.array) or each pair of columns (pandas.DataFrame).
        y: an optional 1d numerical array. If x is 1d and y is given, it computes
          the coefficient between x and y.
        internal_n_clusters: this parameter can be an integer (the maximum number
          of clusters used to split x and y, starting from k=2) or a list of
          integer values (a custom list of k values).
        return_parts: if True, for each object pair, it returns the partitions
          that maximized the coefficient.
        n_chunks_threads_ratio: allows to modify how pairwise comparisons are
          split across different threads. It's given as the ratio parameter of
          function get_chunks.
        n_jobs: number of CPU cores/threads to use for parallelization. The value
          None will use all available cores (`os.cpu_count()`), and negative
          values will use `os.cpu_count() + n_jobs` (exception will be raised
          if this expression yields a result less than 1). Default is 1.
        pvalue_n_perms: if given, it computes the p-value of the
            coefficient using the given number of permutations.
        partitioning_executor: Executor type used for partitioning the data. It
            can be either "thread" (default) or "process". If "thread", it will use
            ThreadPoolExecutor for parallelization, which uses less memory. If
            "process", it will use ProcessPoolExecutor, which might be faster. If
            anything else, it will not parallelize the partitioning step.


    Returns:
        If returns_parts is True, then it returns a tuple with three values:
        1) the coefficients, 2) the partitions indexes that maximized the coefficient
        for each object pair, and 3) the partitions for all objects.
        If return_parts is False, only CCC values are returned.

        cm_values: if x is 2d np.array with x.shape[0] > 2, then cm_values is a 1d
            condensed array of pairwise coefficients. It has size (n * (n - 1)) / 2,
            where n is the number of rows in x. If x and y are given, and they are 1d,
            then cm_values is a scalar. The CCC is always between 0 and 1 (inclusive). If
            any of the two variables being compared has no variation (all values are the
            same), the coefficient is not defined (np.nan). If pvalue_n_permutations is
            an integer greater than 0, then cm_vlaues is a tuple with two elements:
            the first element are the CCC values, and the second element are the p-values
            using pvalue_n_permutations permutations.

        max_parts: an array with n * (n - 1)) / 2 rows (one for each object
            pair) and two columns. It has the indexes pointing to each object's
            partition (parts, see below) that maximized the ARI. If
            cm_values[idx] is nan, then max_parts[idx] will be meaningless.

        parts: a 3d array that contains all the internal partitions generated
            for each object in data. parts[i] has the partitions for object i,
            whereas parts[i,j] has the partition j generated for object i. The
            third dimension is the number of columns in x (if 2d) or elements in
            x/y (if 1d). For example, if you want to access the pair of
            partitions that maximized the CCC given x and y
            (a pair of objects), then max_parts[0] and max_parts[1] have the
            partition indexes in parts, respectively: parts[0][max_parts[0]]
            points to the partition for x, and parts[1][max_parts[1]] points to
            the partition for y. Values could be negative in case
            singleton cases were found (-1; usually because input data has all the same
            value) or for categorical features (-2).
    """
    n_objects = None
    n_features = None
    # this is a boolean array of size n_features with True if the feature is numerical and False otherwise
    X_numerical_type = None
    if x.ndim == 1 and (y is not None and y.ndim == 1):
        # both x and y are 1d arrays
        if not x.shape == y.shape:
            raise ValueError("x and y need to be of the same size")
        n_objects = x.shape[0]
        n_features = 2

        X = np.zeros((n_features, n_objects))
        X_numerical_type = np.full((n_features,), True, dtype=bool)

        X[0, :], X_numerical_type[0] = get_feature_type_and_encode(x)
        X[1, :], X_numerical_type[1] = get_feature_type_and_encode(y)
    elif x.ndim == 2 and y is None:
        # x is a 2d array; two things could happen: 1) this is an numpy array,
        # in that case, features are in rows, objects are in columns; 2) or this is a
        # pandas dataframe, which is the opposite (features in columns and objects in rows),
        # plus we have the features data type (numerical, categorical, etc)

        if isinstance(x, np.ndarray):
            if not get_feature_type_and_encode(x[0, :])[1]:
                raise ValueError("If data is a 2d numpy array, it has to be numerical. Use pandas.DataFrame if "
                                 "you need to mix features with different data types")
            n_objects = x.shape[1]
            n_features = x.shape[0]

            X = x
            X_numerical_type = np.full((n_features,), True, dtype=bool)
        elif hasattr(x, "to_numpy"):
            # Here I assume that if x has the attribute "to_numpy" is of type pandas.DataFrame
            # Using isinstance(x, pandas.DataFrame) would be more appropriate, but I dont want to
            # have pandas as a dependency just for that
            n_objects = x.shape[0]
            n_features = x.shape[1]

            X = np.zeros((n_features, n_objects))
            X_numerical_type = np.full((n_features,), True, dtype=bool)

            for f_idx in range(n_features):
                X[f_idx, :], X_numerical_type[f_idx] = get_feature_type_and_encode(
                    x.iloc[:, f_idx]
                )
    else:
        raise ValueError("Wrong combination of parameters x and y")

    # get number of cores to use
    n_workers = get_n_workers(n_jobs)

    # Converts internal_n_clusters to a list of integers if it's provided.
    if internal_n_clusters is not None:
        _tmp_list = List()

        if isinstance(internal_n_clusters, int):
            # this interprets internal_n_clusters as the maximum k
            internal_n_clusters = range(2, internal_n_clusters + 1)

        for x in internal_n_clusters:
            _tmp_list.append(x)
        internal_n_clusters = _tmp_list

    # get matrix of partitions for each object pair
    range_n_clusters = get_range_n_clusters(n_objects, internal_n_clusters)

    if range_n_clusters.shape[0] == 0:
        raise ValueError(f"Data has too few objects: {n_objects}")

    # store a set of partitions per row (object) in X as a multidimensional
    # array, where the second dimension is the number of partitions per object.
    parts = (
        np.zeros((n_features, range_n_clusters.shape[0], n_objects), dtype=np.int16) - 1
    )

    # cm_values stores the CCC coefficients
    n_features_comp = (n_features * (n_features - 1)) // 2
    cm_values = np.full(n_features_comp, np.nan)
    cm_pvalues = np.full(n_features_comp, np.nan)

    # for each object pair being compared, max_parts has the indexes of the
    # partitions that maximimized the ARI
    max_parts = np.zeros((n_features_comp, 2), dtype=np.uint64)

    with (
        ThreadPoolExecutor(max_workers=n_workers) as executor,
        ProcessPoolExecutor(max_workers=n_workers) as pexecutor,
    ):
        map_func = map
        if n_workers > 1:
            if partitioning_executor == "thread":
                map_func = executor.map
            elif partitioning_executor == "process":
                map_func = pexecutor.map

        # pre-compute the internal partitions for each object in parallel

        # first, create a list with features-k pairs that will be used to parallelize
        # the partitioning step
        inputs = get_chunks(
            [
                (f_idx, c_idx, c)
                for f_idx in range(n_features)
                for c_idx, c in enumerate(range_n_clusters)
            ],
            n_workers,
            n_chunks_threads_ratio,
        )

        # then, flatten the list of features-k pairs into a list that is divided into
        # chunks that will be used to parallelize the partitioning step.
        inputs = [
            [
                (
                    feature_k_pair,
                    X[feature_k_pair[0]],
                    X_numerical_type[feature_k_pair[0]],
                )
                for feature_k_pair in chunk
            ]
            for chunk in inputs
        ]

        for params, ps in zip(inputs, map_func(get_feature_parts, inputs)):
            # get the set of feature indexes and cluster indexes
            f_idxs = [p[0][0] for p in params]
            c_idxs = [p[0][1] for p in params]

            # update the partitions for each feature-k pair
            parts[f_idxs, c_idxs] = ps

        # Below, there are two layers of parallelism: 1) parallel execution
        # across feature pairs and 2) the cdist_parts_parallel function, which
        # also runs several threads to compare partitions using ari. In 2) we
        # need to disable parallelization in case len(cm_values) > 1 (that is,
        # we have several feature pairs to compare), because parallelization is
        # already performed at this level. Otherwise, more threads than
        # specified by the user are started.
        map_func = map
        cdist_executor = False
        inner_executor = DummyExecutor()

        if n_workers > 1:
            if n_features_comp == 1:
                map_func = map
                cdist_executor = executor
                inner_executor = pexecutor

            else:
                map_func = pexecutor.map

        # iterate over all chunks of object pairs and compute the coefficient
        inputs = get_chunks(n_features_comp, n_workers, n_chunks_threads_ratio)
        inputs = [
            (
                i,
                n_features,
                parts,
                pvalue_n_perms,
                n_workers,
                n_chunks_threads_ratio,
                cdist_executor,
                inner_executor,
            )
            for i in inputs
        ]

        for params, (max_ari_list, max_part_idx_list, pvalues) in zip(
            inputs, map_func(compute_coef, inputs)
        ):
            f_idx = params[0]

            cm_values[f_idx] = max_ari_list
            max_parts[f_idx, :] = max_part_idx_list
            cm_pvalues[f_idx] = pvalues

    print("CPU parts:")
    print(parts)
    # return an array of values or a single scalar, depending on the input data
    if cm_values.shape[0] == 1:
        if return_parts:
            if pvalue_n_perms is not None and pvalue_n_perms > 0:
                return (cm_values[0], cm_pvalues[0]), max_parts[0], parts
            else:
                return cm_values[0], max_parts[0], parts
        else:
            if pvalue_n_perms is not None and pvalue_n_perms > 0:
                return cm_values[0], cm_pvalues[0]
            else:
                return cm_values[0]

    if return_parts:
        if pvalue_n_perms is not None and pvalue_n_perms > 0:
            return (cm_values, cm_pvalues), max_parts, parts
        else:
            return cm_values, max_parts, parts
    else:
        if pvalue_n_perms is not None and pvalue_n_perms > 0:
            return cm_values, cm_pvalues
        else:
            return cm_values
