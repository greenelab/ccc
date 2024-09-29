import numpy as np
import cupy as cp
from numba import njit
from numba import cuda
import rmm


d_unravel_index_str = """
/**
 * @brief Unravel a flat index to the corresponding 2D indicis
 * @param[in] flat_idx The flat index to unravel
 * @param[in] num_cols Number of columns in the 2D array
 * @param[out] row Pointer to the row index
 * @param[out] col Pointer to the column index
 */
extern "C" __device__ __host__ inline void unravel_index(size_t flat_idx, size_t num_cols, size_t* row, size_t* col) {
    *row = flat_idx / num_cols;  // Compute row index
    *col = flat_idx % num_cols;  // Compute column index
}

"""

d_get_coords_from_index_str = """
#include <math.h>
extern "C" __device__ __host__ inline void get_coords_from_index(int n_obj, int idx, int* x, int* y) {
    // Calculate 'b' based on the input n_obj
    int b = 1 - 2 * n_obj;
    // Calculate 'x' using the quadratic formula part
    float discriminant = b * b - 8 * idx;
    float x_float = floor((-b - sqrt(discriminant)) / 2);
    // Assign the integer part of 'x'
    *x = static_cast<int>(x_float);
    // Calculate 'y' based on 'x' and the index
    *y = static_cast<int>(idx + (*x) * (b + (*x) + 2) / 2 + 1);
}

"""

k_ari_str = """
/**
 * @brief Main ARI kernel. Now only compare a pair of ARIs
 * @param n_parts Number of partitions of each feature
 * @param n_objs Number of objects in each partitions
 * @param n_part_mat_elems Number of elements in the square partition matrix
 * @param n_elems_per_feat Number of elements for each feature, i.e., part[i].x * part[i].y
 */
extern "C" __global__
void ari(const int4* parts,
         const int4* uniqs,
         const int n_aris,
         const int n_parts,
         const int n_objs,
         const uint32 n_elems_per_feat,
         const int n_part_mat_elems,
         float* out)
         )
{
    // tid corresponds to the ari idx
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // used for max reduction
    // int part_part_elems = n_parts * n_parts;
    
    // obtain the corresponding parts and unique counts
    int feature_comp_flat_idx = tid / n_part_mat_elems;   // comparison pair index for two features
    int part_pair_flat_idx = tid % part_part_elems;  // comparison pair index for two partitions of one feature pair
    int i, j;
    // unravel the feature indices
    get_coords_from_index(n_parts, feature_comp_flat_idx, &i, &j);
    // unravel the partition indices
    int m, n;
    unravel_index(part_pair_flat_idx, n_parts, &m, &n);
    
    // Make pointers to select the parts and unique counts for the feature pair
    int4* t_data_parti = parts + i * n_elems_per_feat + m * n_objs ;  // t_ for thread
    int4* t_data_partj = parts + j * n_elems_per_feat + n * n_objs ;
    int4* t_data_uniqi = uniqs + i * n_parts + m;
    int4* t_data_uniqj = uniqs + j * n_parts + n;
    
    // Load gmem data into smem by using different threads
    
    
    
    // Initialize shared memory
    int part_mat_first_tid = tid * part_part_elems;
    __syncthreads();
}

"""


def get_kernel():
    """
    Kernel to compute the air between two partitions indexed from the 3D input array parts.

    The first thread of each logical part vs part ari matrix is responsible to reduce the matrix to the max ari.
    See the document for illustrations.

    raw kernel args:
        parts: 3D device array with cluster assignments for x features, y partitions, and z objects.
        uniqs: 2D device array with the number of unique elements for feature x and partition y.
        n_aris: Number of ARI computations to perform.
        n_parts: Number of partitions of a feature, i.e., len(n_range_clusters) to compare.
        out: Pointer to the pre-allocated 1D device output array with length of number of features to compare.
    """

    cuda_code = d_get_coords_from_index_str + k_ari_str

    kernel = cp.RawKernel(code=cuda_code, backend="nvcc").get_function("ari")
    return kernel


def adjusted_rand_index(
                        part0: np.ndarray,
                        part1: np.ndarray,
                        size: int,
                        n_uniq0: int,
                        n_uniq1: int,
                        out: np.ndarray,
                        compare_pair_id: int,
                        i: int,
                        j: int,
                        stream: cp.cuda.Stream = None):
    """
    Computes the adjusted Rand index (ARI) between two clustering partitions.
    The code is based on the sklearn implementation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    See copyright notice at the top of this file.

    Host function to coordinate the GPU kernel.

    Args:
        part0: a 1d array with cluster assignments for n objects.
        part1: a 1d array with cluster assignments for n objects.
        size: the number of objects in the partitions.
        n_uniq0: the number of unique elements in part0.
        n_uniq1: the number of unique elements in part1.
        out: pointer to the output array containing all the ARI values. # TODO: make local
        compare_pair_id: the index of the pair of partitions to compare.
        i: the index of the first partition.
        j: the index of the second partition.
        stream: the CUDA stream to use.

    Returns:
        A number representing the adjusted Rand index between two clustering
        partitions. This number is between something around 0 (partitions do not
        match; it could be negative in some cases) and 1.0 (perfect match).
    """
    # TODO:
    # Implement numpy ravel in the kernel using shared memory?
    # Use different streams for different pairs?
    # Ref api: CUML confusion_matrix
    if not size >= 2:
        raise ValueError("Need at least two samples to compare.")



    (tn, fp), (fn, tp) = get_pair_confusion_matrix(part0, part1)
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        res = 1.0

    res = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    out[compare_pair_id, i, j] = res


def ari_dim2(parts: cp.ndarray,
             n_partitions: int,
             n_features_comp: int,
             out: cp.ndarray,
             unique_element_counts: cp.ndarray):
    """
    Function to compute the ARI between partitions on the GPU. This function is responsible for launching the kernel
    in different streams for each pair of partitions.

    Args:
        parts: 3D device array with cluster assignments for x features, y partitions, and z objects.
        Example initialization for this array: d_parts = cp.empty((nx, ny, nz), dtype=np.int16) - 1

        n_partitions: Number of partitions of a feature to compare.

        n_features_comp: Pre-computed number of features to compare.

        out: Pointer to the pre-allocated 1D device output array with length of n_features_comp.

        unique_element_counts: 2D device array with the number of unique elements for feature x and partition y.
    """

    # Can use non-blocking CPU scheduling or CUDA dynamic parallelism to launch the kernel for each pair of partitions.

    # Each kernel launch will be responsible for computing the ARI between two partitions.
    n_part_mat_elems = n_partitions * n_partitions
    # Each thread
    n_ari_pairs = n_partitions * n_part_mat_elems
    cm_values = cp.full(n_features_comp, cp.nan)
    # Todo: how many ari pairs? n_range_cluster?
    threads_per_block = 1
    blocks_per_grid = (n_ari_pairs + threads_per_block - 1) // threads_per_block
    ari_kernel = get_kernel()
    # Todo: use different streams
    ari_kernel(grid=(blocks_per_grid,),
               block=(threads_per_block,),
               args=(parts, unique_element_counts, n_features_comp, n_part_mat_elems, out))

    raise NotImplementedError("Not implemented yet")
