"""
Contains function that implement the Clustermatch Correlation Coefficient (CCC).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Union

import numpy as np
from numpy.typing import NDArray
from numba import njit, get_num_threads
from numba.typed import List

from ccc.pytorch.core import unravel_index_2d
from ccc.sklearn.metrics import adjusted_rand_index as ari
from ccc.scipy.stats import rank
from ccc.utils import chunker


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


@njit(cache=True, nogil=True)
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

    percentiles = [0.0] + get_perc_from_k(k) + [1.0]

    cut_points = np.searchsorted(data_perc[data_sorted], percentiles, side="right")

    current_cluster = 0
    part = np.zeros(data.shape, dtype=np.int16) - 1

    for i in range(len(cut_points) - 1):
        lim1 = cut_points[i]
        lim2 = cut_points[i + 1]

        part[data_sorted[lim1:lim2]] = current_cluster
        current_cluster += 1

    return part


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def get_parts(
    data: NDArray, range_n_clusters: tuple[int], data_is_numerical=True
) -> NDArray[np.int16]:
    """
    Given a 1d data array, it computes a partition for each k value in the given
    range of clusters. This function only supports numerical data, and it
    always runs run_run_quantile_clustering with the different k values.
    If partitions with only one cluster are returned (singletons), then the
    returned array will have negative values.

    Args:
        data: a 1d data vector. It is assumed that there are no nans.
        range_n_clusters: a tuple with the number of clusters.
        data_type: "numerical" or "categorical"

    Returns:
        A numpy array with shape (number of clusters, data rows) with
        partitions of data.
    """
    parts = np.zeros((len(range_n_clusters), data.shape[0]), dtype=np.int16) - 1

    if data_is_numerical:
        for idx in range(len(range_n_clusters)):
            k = range_n_clusters[idx]
            parts[idx] = run_quantile_clustering(data, k)

        # remove singletons
        partitions_ks = np.array([len(np.unique(p)) for p in parts])
        parts[partitions_ks == 1, :] = -1
    else:
        # if the data is categorical, then the encoded feature is already the partition
        parts[0] = data.astype(np.int16)

    return parts


def cdist_parts_basic(x: NDArray, y: NDArray) -> NDArray[float]:
    """
    It implements the same functionality in scipy.spatial.distance.cdist but
    for clustering partitions, and instead of a distance it returns the adjusted
    Rand index (ARI). In other words, it mimics this function call:

        cdist(x, y, metric=ari)

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
        for j in range(res.shape[1]):
            if x[i, 0] >= 0 and y[j, 0] >= 0:
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


def to_numpy(x):
    """
    Converts x into a numpy array. It is used to convert pandas Series and
    DataFrames into numpy objects.
    """
    if x is None:
        return x

    func = getattr(x, "to_numpy", None)
    if not callable(func):
        return x

    return x.to_numpy()


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


def get_feature_type_and_encode(feature_data):
    """
    Given the data of one feature as a 1d numpy array (it could also be a pandas.Series),
    it returns the same data if it is numerical (float, signed or unsigned integer) or an
    encoded version if it is categorical (each category value has a unique integer starting from
    zero).
    """
    data_type_is_numerical = feature_data.dtype.kind in ("f", "i", "u")
    if data_type_is_numerical:
        return feature_data, data_type_is_numerical

    # here np.unique with return_inverse encodes categorical values into numerical ones
    return np.unique(feature_data, return_inverse=True)[1], data_type_is_numerical


def ccc(
    x: NDArray,
    y: NDArray = None,
    internal_n_clusters: Union[int, Iterable[int]] = None,
    return_parts: bool = False,
    n_chunks_threads_ratio: int = 3,
) -> tuple[NDArray[float], NDArray[np.uint64], NDArray[np.int16]]:
    """
    This is the main function that computes the Clustermatch Correlation
    Coefficient (CCC) between two arrays. This implementation only supports
    numerical data for optimization purposes, but the original implementation
    can also work with categorical data (https://github.com/sinc-lab/clustermatch).

    To control the number of threads used, set the NUMBA_NUM_THREADS variable
    to an integer. For example, NUMBA_NUM_THREADS=2 will use 2 cores.

    Args:
        x: an 1d or 2d numerical array with the data. NaN are not supported.
          If it is 2d, then the coefficient is computed for each pair of rows.
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
            the partition for y.
            TODO: mention here that "invalid" or "missing" partitions have all -1 values
              for example, for categorical values only the first one makes sense
    """
    n_objects = None
    n_features = None
    # this is a boolean array with 1 if the feature is numerical and 0 otherwise
    X_numerical_type = None
    if x.ndim == 1 and (y is not None and y.ndim == 1):
        # both x and y are 1d arrays
        assert x.shape == y.shape, "x and y need to be of the same size"
        n_objects = x.shape[0]
        n_features = 2

        X = np.zeros((n_features, n_objects))
        X_numerical_type = np.full((n_features,), True, dtype=bool)
        # for idx in range(n_features):
        #     feature_data, feature_is_numerical = get_feature_type_and_encode()

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

    # get number of cores to use
    # FIXME: this is not necessary, use parameter like n_jobs instead
    #  and force numba to use the same number of n_jobs (paramter)
    default_n_threads = get_num_threads()

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
    # n = features_list.shape[0]
    n_features_comp = (n_features * (n_features - 1)) // 2
    cm_values = np.full(n_features_comp, np.nan)

    # for each object pair being compared, max_parts has the indexes of the
    # partitions that maximimized the ARI
    max_parts = np.zeros((n_features_comp, 2), dtype=np.uint64)

    with ThreadPoolExecutor(max_workers=default_n_threads) as executor:
        # pre-compute the internal partitions for each object in parallel
        inputs = get_chunks(n_features, default_n_threads, n_chunks_threads_ratio)

        def compute_parts(idxs):
            return np.array(
                [get_parts(X[i], range_n_clusters, X_numerical_type[i]) for i in idxs]
            )

        for idx, ps in zip(inputs, executor.map(compute_parts, inputs)):
            parts[idx] = ps

        # Below, there are two layers of parallelism: 1) parallel execution
        # across object pairs and 2) the cdist_parts_parallel function, which
        # also runs several threads to compare partitions using ari. In 2) we
        # need to disable parallelization in case len(cm_values) > 1 (that is,
        # we have several object pairs to compare), because parallelization is
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
                # (negative values)
                if obji_parts[0, 0] < 0 or objj_parts[0, 0] < 0:
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
