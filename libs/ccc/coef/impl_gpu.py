"""
This module contains the CUDA implementation of the CCC
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Iterable, Union, List, Tuple

import numpy as np
import cupy as cp
from numpy.typing import NDArray
from numba import njit
from numba import cuda
from fractions import Fraction

from ccc.pytorch.core import unravel_index_2d
from ccc.scipy.stats import rank
from ccc.sklearn.metrics import adjusted_rand_index as ari
from ccc.utils import chunker

# @njit(cache=True, nogil=True)
def get_perc_from_k(k: int) -> NDArray[np.float64]:
    """
    It returns the percentiles (from 0.0 to 1.0) that separate the data into k
    clusters. For example, if k=2, it returns [0.5]; if k=4, it returns [0.25,
    0.50, 0.75].

    Args:
        k: number of clusters. If less than 2, the function returns an empty
            list.

    Returns:
        A numpy array of percentiles (from 0.0 to 1.0).
    """
    np.set_printoptions(precision=17)
    if k < 2:
        return np.array([], dtype='float64')
    return np.linspace(1/k, 1-1/k, k-1, dtype='float64')


# @njit(cache=True, nogil=True)
def get_range_n_percentages(ks: NDArray[np.uint8], as_percentage: bool = False) -> NDArray[float]:
    """
    It returns lists of the percentiles (from 0.0 to 1.0) that separate the data into k[i] clusters
    
    Args:
        ks: an array of numbers of clusters.
    
    Returns:
        A 2D sparse matrix of percentiles (from 0.0 to 1.0).
    """
    # Todo: research on if numba can optimize this
    # Emtpy & null check
    if ks.size == 0:
        return np.empty((0, 0), dtype=float)
    # Number of rows of the returning matrix
    n_rows = len(ks)
    # Number of columns of the returning matrix, dominated by the largest k, which specifies the # of clusters
    n_cols = np.max(ks) - 1
    percentiles = np.full((n_rows, n_cols), np.nan, dtype=float)
    for idx, k in enumerate(ks):
        perc = get_perc_from_k(k)
        if as_percentage:
            perc = np.round(perc * 100).astype(float)  # Convert to percentage and round
        percentiles[idx, :len(perc)] = perc
    return percentiles


def get_feature_type_and_encode(feature_data: NDArray) -> tuple[NDArray, bool]:
    """
    Given the data of one feature as a 1d numpy array (it could also be a pandas.Series),
    it returns the same data if it is numerical (float, signed or unsigned integer) or an
    encoded version if it is categorical (each category value has a unique integer starting from
    zero).` f

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


# @njit(cache=True, nogil=True)
def get_range_n_clusters(
        n_items: int, internal_n_clusters: Iterable[int] = None
) -> NDArray[np.uint8]:
    """
    Given the number of features it returns a tuple of k values to cluster those
    features into. By default, it generates a tuple of k values from 2 to
    int(np.round(np.sqrt(n_items))) (inclusive). For example, for 25 features,
    it will generate this array: (2, 3, 4, 5).

    Args:
        n_items: a positive number representing the number of features that
            will be clustered into different groups/clusters.
        internal_n_clusters: it allows to force a different list of clusters. It
            must be a list of integers. Repeated or invalid values will be dropped,
            such as values lesser than 2 (a singleton partition is not allowed).

    Returns:
        A numpy array with integer values representing numbers of clusters.
    """

    if internal_n_clusters:
        # remove k values that are invalid
        clusters_range_list = list(
            set([int(x) for x in internal_n_clusters if 1 < x < n_items])
        )
    else:
        # default behavior if no internal_n_clusters is given: return range from
        # 2 to sqrt(n_items)
        n_sqrt = int(np.round(np.sqrt(n_items)))
        n_sqrt = min((n_sqrt, 10))
        clusters_range_list = list(range(2, n_sqrt + 1))

    return np.array(clusters_range_list, dtype=np.uint16)


# # Todo: restore the original implementation
# @cuda.jit(device=True)
# def get_coords_from_index(n_obj: int, idx: int) -> tuple[int]:
#     """
#     Given the number of objects and an index, it returns the row/column
#     position of the pairwise matrix. For example, if there are n_obj objects
#     (such as genes), a condensed 1d array can be created with pairwise
#     comparisons between genes, as well as a squared symmetric matrix. This
#     function receives the number of objects and the index of the condensed
#     array, and returns the coordinates of the squared symmetric matrix.
#     Args:
#         n_obj: the number of objects.
#         idx: the index of the condensed pairwise array across all n_obj objects.
#     Returns
#         A tuple (i, j) with the coordinates of the squared symmetric matrix
#         equivalent to the condensed array.
#     """
#     b = 1 - 2 * n_obj
#     x = math.floor((-b - math.sqrt(b**2 - 8 * idx)) / 2)
#     y = idx + x * (b + x + 2) / 2 + 1
#     return int(x), int(y)


@njit(cache=True, nogil=True)
def get_coords_from_index(n_obj: int, idx: int) -> tuple[int]:
    """
    Given the number of objects and and index, it returns the row/column
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


def convert_n_clusters(internal_n_clusters: Optional[Union[int, List[int]]]) -> List[int]:
    if internal_n_clusters is None:
        return []

    if isinstance(internal_n_clusters, int):
        return list(range(2, internal_n_clusters + 1))

    return list(internal_n_clusters)


def get_parts(X: NDArray,
              range_n_clusters: NDArray[np.uint8],
              data_is_numerical: bool = True
              ) -> cp.ndarray:
    """
    Compute parts using CuPy for GPU acceleration.

    Parameters:
    X: Input data array of shape (n_features, n_objects)
    range_n_clusters: Array of cluster numbers
    range_n_percentages: Array of percentages for each cluster number

    Returns:
    Reference to the computed partitions on the device global memory
    """

    # Handle case when X is a 1D array
    if X.ndim == 1:
        nx = 1 # n_features
        ny = range_n_clusters.shape[0] # n_clusters
        nz = X.shape[0] # n_objects
    else:
        nx = X.shape[0] # n_features
        ny = range_n_clusters.shape[0] # n_clusters
        nz = X.shape[1] # n_objects
    # print(f"{nx}, {ny}, {nz}")

    # Allocate arrays on device global memory
    d_parts = cp.empty((nx, ny, nz), dtype=np.int16) - 1
    # print(f"prev parts: {d_parts}")

    if data_is_numerical:
        # Transfer data to device
        d_X = cp.asarray(X)
        # Get cutting percentages for each cluster
        range_n_percentages = get_range_n_percentages(range_n_clusters)
        d_range_n_percentages = cp.asarray(range_n_percentages, dtype=float)

        for x in range(nx):
            for y in range(ny):
                objects = d_X[x, :] if d_X.ndim == 2 else d_X # objects in a feature row
                percentages = d_range_n_percentages[y, :]
                print(f"GPU percentiles: {percentages}")
                bins = cp.quantile(objects, percentages)
                print(f"GPU quantiles: {bins}")
                partition = cp.digitize(objects, bins, right=True)
                d_parts[x, y, :] = partition

        # Remove singletons by putting -2 as values
        partitions_ks = cp.array([len(cp.unique(d_parts[i, j, :])) for i in range(nx) for j in range(ny)]).reshape(nx, ny)
        d_parts[partitions_ks == 1] = -2
    else:
        # If the data is categorical, then the encoded feature is already the partition
        # Only the first partition is filled, the rest will be -1 (missing)
        d_parts[:, 0] = cp.asarray(X.astype(cp.int16))

    # Move data back to host
    # h_parts = cp.asnumpy(d_parts)
    # print(f"after parts: {d_parts}")
    cp.cuda.runtime.deviceSynchronize()
    return d_parts


# # Todo: kernel on partition paris (1D, 1D) instead of paris of matrices (2D, 2D)
# @cuda.jit(device=True)
# def cdist_parts_basic(x: NDArray, y: NDArray, out: NDArray, compare_pair_id: int) -> None:
#     """
#     It implements the same functionality in scipy.spatial.distance.cdist but
#     for clustering partitions, and instead of a distance it returns the adjusted
#     Rand index (ARI). In other words, it mimics this function call:
#
#         cdist(x, y, metric=ari)
#
#     Only partitions with positive labels (> 0) are compared. This means that
#     partitions marked as "singleton" or "empty" (categorical data) are not
#     compared. This has the effect of leaving an ARI of 0.0 (zero).
#
#     Args:
#         x: a 2d array with m_x clustering partitions in rows and n objects in
#           columns.
#         y: a 2d array with m_y clustering partitions in rows and n objects in
#           columns.
#
#     Returns:
#         A 2d array with m_x rows and m_y columns and the ARI between each
#         partition pair. Each ij entry is equal to ari(x[i], y[j]) for each i
#         and j.
#     """
#
#     for i in range(out.shape[0]):
#         if x[i, 0] < 0:
#             continue
#
#         for j in range(out.shape[1]):
#             if y[j, 0] < 0:
#                 continue
#
#             # res[i, j] = ari(x[i], y[j])
#             # ari(x[i], y[j], out, compare_pair_id, i, j)
#             res = ari(x[i], y[j])
#             print(res)
#
#     return
#
#
# @cuda.jit
# def compute_coef(
#                 parts: cuda.cudadrv.devicearray,
#                 max_ari_list: cuda.cudadrv.devicearray,
#                 max_part_idx_list: cuda.cudadrv.devicearray,
#                 temp_outs: cuda.cudadrv.devicearray,
#                 compare_pair_id: int,
# ):
#     """
#     Given an index representing each a pair of
#     objects/rows/genes, it computes the CCC coefficient for
#     each of them.
#
#     Args:
#         parts: A reference to the 3d GPU partitions array.
#         max_part_idx_list: A reference to the 2d GPU array that stores the indexes of the partitions that maximized the ARI.
#         max_ari_list: A reference to the 1d GPU array that stores the maximum ARI values.
#         compare_pair_id: An id representing a pair of partitions to be compared.
#
#     Returns:
#         Returns a tuple with two arrays. These two arrays are the same
#           arrays returned by the main cm function (cm_values and
#           max_parts) but for a subset of the data.
#     """
#     n_features = parts.shape[0]
#
#     # for idx, data_idx in enumerate(compare_pair_id):
#     i, j = get_coords_from_index(n_features, compare_pair_id)
#
#     # get partitions for the pair of objects
#     obji_parts, objj_parts = parts[i], parts[j]
#
#     # compute ari only if partitions are not marked as "missing"
#     # (negative values), which is assigned when partitions have
#     # one cluster (usually when all data in the feature has the same
#     # value).
#     if obji_parts[0, 0] == -2 or objj_parts[0, 0] == -2:
#         return
#
#     # compare all partitions of one object to the all the partitions
#     # of the other object, and get the maximium ARI
#
#     cdist_parts_basic(
#         obji_parts,
#         objj_parts,
#         temp_outs,
#         compare_pair_id,
#     )
#     # max_flat_idx = comp_values.argmax()
#
#     # max_idx = unravel_index_2d(max_flat_idx, comp_values.shape)
#     # max_part_idx_list[compare_pair_id] = max_idx
#     # max_ari_list[compare_pair_id] = np.max((comp_values[max_idx], 0.0))
#     #
#     # return max_ari_list, max_part_idx_list
#     return

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
        # Got two chunks
        new_chunk = get_chunks(res[idx], 2)
        res[idx] = new_chunk[0]
        # Insert the second chunk in the next position
        res.insert(idx + 1, new_chunk[1])

    return res


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
        executor: an pool executor where jobs will be submitted.

    Results:
        Same as in cdist_parts_basic.
    """
    res = np.zeros((x.shape[0], y.shape[0]))

    inputs = list(chunker(np.arange(res.shape[0]), 1))

    tasks = {executor.submit(cdist_parts_basic, x[idxs], y): idxs for idxs in inputs}
    for t in as_completed(tasks):
        idx = tasks[t]
        res[idx, :] = t.result()

    return res


def ccc(
        x: NDArray,
        y: NDArray = None,
        internal_n_clusters: Union[int, Iterable[int]] = None,
        return_parts: bool = False,
        n_chunks_threads_ratio: int = 1,
        n_jobs: int = 1,
) -> tuple[NDArray[float], NDArray[np.uint64], NDArray[np.int16]]:
    """
    This is the main function that computes the Clustermatch Correlation
    Coefficient (CCC) between two arrays. The implementation supports numerical
    and categorical data.

    Args:
        x: an 1d or 2d numerical array with the data. NaN are not supported.
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
        n_jobs: number of CPU cores to use for parallelization. The value
          None will use all available cores (`os.cpu_count()`), and negative
          values will use `os.cpu_count() - n_jobs`. Default is 1.

    Returns:
        If return_parts is False, only CCC values are returned.
        In that case, if x is 2d, a np.ndarray of size n x n is
        returned with the coefficient values, where n is the number of rows in x.
        If only a single coefficient was computed (for example, x and y were
        given as 1d arrays each), then a single scalar is returned.

        If returns_parts is True, then it returns a tuple with three values:
        1) the
        coefficients, 2) the partitions indexes that maximized the coefficient
        for each object pair, and 3) the partitions for all objects.

        cm_values: if x is 2d, then it is a 1d condensed array of pairwise
            coefficients. It has size (n * (n - 1)) / 2, where n is the number
            of rows in x. If x and y are given, and they are 1d, then this is a
            scalar. The CCC is always between 0 and 1
            (inclusive). If any of the two variables being compared has no
            variation (all values are the same), the coefficient is not defined
            (np.nan).

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
        # Create a matrix to store both x and y
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
            assert get_feature_type_and_encode(x[0, :])[1], (
                "If data is a 2d numpy array, it has to be numerical. Use pandas.DataFrame if "
                "you need to mix features with different data types"
            )
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

            for idx in range(n_features):
                X[idx, :], X_numerical_type[idx] = get_feature_type_and_encode(
                    x.iloc[:, idx]
                )
    else:
        raise ValueError("Wrong combination of parameters x and y")

    # 1. Partitions Computation

    # Converts internal_n_clusters to a list of integers if it's provided.
    internal_n_clusters = convert_n_clusters(internal_n_clusters)

    # Get matrix of partitions for each object pair
    range_n_clusters = get_range_n_clusters(n_objects, internal_n_clusters)

    if range_n_clusters.shape[0] == 0:
        raise ValueError(f"Data has too few objects: {n_objects}")

    # cm_values stores the CCC coefficients
    n_features_comp = (n_features * (n_features - 1)) // 2
    cm_values = np.full(n_features_comp, np.nan)

    # for each object pair being compared, max_parts has the indexes of the
    # partitions that maximimized the ARI
    max_parts = np.zeros((n_features_comp, 2), dtype=np.uint64)

    # X here (and following) is a numpy array features are in rows, objects are in columns

    # Compute partitions for each feature using CuPy
    d_parts = get_parts(X, range_n_clusters)
    print("GPU parts:")
    print(d_parts)

    # 2. CCC coefficient computation

    # # allocate result arrays on device global memory
    # d_max_ari_list = cp.full(n_features_comp, cp.nan, dtype=float)
    # d_max_part_idx_list = cp.zeros((n_features_comp, 2), dtype=np.uint64)
    # # allocate temporary arrays on device global memory
    # d_outs = cp.empty((n_features_comp, range_n_clusters.shape[0], range_n_clusters.shape[0]), dtype=cp.float32)
    # print(f"before d_outs: {d_outs}")
    # # use 1D gird to parallelize the computation of CCC coefficients
    # # Todo: optimize this using updated c_dist function that only compare one partition at a time
    # threads_per_block = 1
    # blocks_per_grid = n_features_comp
    # for i in range(n_features_comp):
    #     # Directly pass CuPy arrays to kernels JITed with Numba
    #     compute_coef[blocks_per_grid, threads_per_block](d_parts, d_max_ari_list, d_max_part_idx_list, d_outs, i)
    # # Wait for all comparisons to finish
    # cuda.synchronize()
    # print(f"after d_outs: {d_outs}")
    # # Transfer data back to host
    # max_ari_list = cp.asnumpy(d_max_ari_list)
    # max_part_idx_list = cp.asnumpy(d_max_part_idx_list)
    # print(max_ari_list)
    # print(max_part_idx_list)

    # Use CPU multi-threading for baseline
    parts = cp.asnumpy(d_parts)

    default_n_threads = os.cpu_count()

    with ThreadPoolExecutor(max_workers=default_n_threads) as executor:

        # Below, there are two layers of parallelism: 1) parallel execution
        # across feature pairs and 2) the cdist_parts_parallel function, which
        # also runs several threads to compare partitions using ari. In 2) we
        # need to disable parallelization in case len(cm_values) > 1 (that is,
        # we have several feature pairs to compare), because parallelization is
        # already performed at this level. Otherwise, more threads than
        # specified by the user are started.
        cdist_parts_enable_threading = True if n_features_comp == 1 else False

        cdist_func = None
        map_func = executor.map
        if cdist_parts_enable_threading:
            map_func = map

            def cdist_func(x, y):
                return cdist_parts_parallel(x, y, executor)

        else:
            cdist_func = cdist_parts_basic

        # compute coefficients
        def compute_coef(idx_list):
            """
            Given a list of indexes representing each a pair of
            objects/rows/genes, it computes the CCC coefficient for
            each of them. This function is supposed to be used to parallelize
            processing.

            Args:
                idx_list: a list of indexes (integers), each of them
                  representing a pair of objects.

            Returns:
                Returns a tuple with two arrays. These two arrays are the same
                  arrays returned by the main cm function (cm_values and
                  max_parts) but for a subset of the data.
            """
            n_idxs = len(idx_list)
            max_ari_list = np.full(n_idxs, np.nan, dtype=float)
            max_part_idx_list = np.zeros((n_idxs, 2), dtype=np.uint64)

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
                comp_values = cdist_func(
                    obji_parts,
                    objj_parts,
                )
                max_flat_idx = comp_values.argmax()

                max_idx = unravel_index_2d(max_flat_idx, comp_values.shape)
                max_part_idx_list[idx] = max_idx
                max_ari_list[idx] = np.max((comp_values[max_idx], 0.0))

            return max_ari_list, max_part_idx_list

        # iterate over all chunks of object pairs and compute the coefficient
        inputs = get_chunks(n_features_comp, default_n_threads, n_chunks_threads_ratio)

        for idx, (max_ari_list, max_part_idx_list) in zip(
            inputs, map_func(compute_coef, inputs)
        ):
            cm_values[idx] = max_ari_list
            max_parts[idx, :] = max_part_idx_list

    # return an array of values or a single scalar, depending on the input data
    if cm_values.shape[0] == 1:
        if return_parts:
            return cm_values[0], max_parts[0], parts
        else:
            return cm_values[0]

    if return_parts:
        return cm_values, max_parts, parts
    else:
        return cm_values

# Dev notes
# 1. parallelize get_parst
# 1.1 gpu percentile computation
# 1.1 gpu data points binning
#   can be a kernel for-loop to compute parts on different percentile
