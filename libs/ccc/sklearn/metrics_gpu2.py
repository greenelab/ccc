import numpy as np
import cupy as cp
from numba import njit
from numba import cuda
import rmm


d_get_confusion_matrix_str = """
/**
 * @brief CUDA device function to compute the pair confusion matrix
 * @param[in] contingency Pointer to the contingency matrix
 * @param[in] sum_rows Pointer to the sum of rows in the contingency matrix
 * @param[in] sum_cols Pointer to the sum of columns in the contingency matrix
 * @param[in] n_objs Number of objects in each partition
 * @param[in] k Number of clusters (assuming k is the max of clusters in part0 and part1)
 * @param[out] C Pointer to the output pair confusion matrix (2x2)
 */
__device__ void get_pair_confusion_matrix(
    const int* __restrict__ contingency,
    int * sum_rows,
    int * sum_cols,
    const int n_objs,
    const int k,
    int* C
) {
    // Initialize sum_rows and sum_cols
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        sum_rows[i] = 0;
        sum_cols[i] = 0;
    }
    __syncthreads();

    // Compute sum_rows and sum_cols
    for (int i = threadIdx.x; i < k * k; i += blockDim.x) {
        int row = i / k;
        int col = i % k;
        int val = contingency[i];
        atomicAdd(&sum_cols[col], val);
        atomicAdd(&sum_rows[row], val);
    }
    __syncthreads();
    // print sum_rows and sum_cols in arrays for debugging
    if (threadIdx.x == 0) {
        printf("sum_rows:\\n");
        for (int i = 0; i < k; ++i) {
            printf("%d ", sum_rows[i]);
        }
        printf("\\nsum_col:\\n");
        for (int i = 0; i < k; ++i) {
            printf("%d ", sum_cols[i]);
        }
    }

    // Compute sum_squares
    int sum_squares;
    if (threadIdx.x == 0) {
        sum_squares = 0;
        for (int i = 0; i < k * k; ++i) {
            sum_squares += (contingency[i]) * contingency[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        printf("sum_squares: %d\\n", sum_squares);
    }

    // Compute C[1,1], C[0,1], C[1,0], and C[0,0]
    if (threadIdx.x == 0) {
        C[3] = sum_squares - n_objs;  // C[1,1]

        int temp = 0;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                temp += (contingency[i * k + j]) * sum_cols[j];
            }
        }
        C[1] = temp - sum_squares;  // C[0,1]

        temp = 0;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                temp += (contingency[j * k + i]) * sum_rows[j];
            }
        }
        C[2] = temp - sum_squares;  // C[1,0]

        C[0] = n_objs * n_objs - C[1] - C[2] - sum_squares;  // C[0,0]

        // print C
        printf("C[0,0]: %d, C[0,1]: %d, C[1,0]: %d, C[1,1]: %d\\n", C[0], C[1], C[2], C[3]);
        
        // compute ARI
        int tn = static_cast<float>(C[0]);
        int fp = static_cast<float>(C[1]);
        int fn = static_cast<float>(C[2]);
        int tp = static_cast<float>(C[3]);
        printf("tn: %d, fp: %d, fn: %d, tp: %d\\n", tn, fp, fn, tp);
        float ari = 0.0;
        if (fn == 0 && fp ==0) {
            ari = 1.0;
        } else {
            ari = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
        }
        printf("ari: %f\\n", ari);
    }
    __syncthreads();
}

"""

d_get_contingency_matrix_str = """
/**
 * @brief Compute the contingency matrix for two partitions using shared memory
 * @param[in] part0 Pointer to the first partition array
 * @param[in] part1 Pointer to the second partition array
 * @param[in] n Number of elements in each partition array
 * @param[out] shared_cont_mat Pointer to shared memory for storing the contingency matrix
 * @param[in] k Maximum number of clusters (size of contingency matrix is k x k)
 */
__device__ void get_contingency_matrix(int* part0, int* part1, int n, int* shared_cont_mat, int k) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_threads = blockDim.x;
    int num_blocks = gridDim.x;

    // Initialize shared memory
    for (int i = tid; i < k * k; i += num_threads) {
        shared_cont_mat[i] = 0;
    }
    __syncthreads();

    // Process elements
    for (int i = tid; i < n; i += num_threads) {
        int row = part0[i];
        int col = part1[i];
        
        if (row < k && col < k) {
            atomicAdd(&shared_cont_mat[row * k + col], 1);
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        for (int i = 0; i < k; ++i)
        {
            printf("\\n");
            for (int j = 0; j < k; ++j)
            {
                printf("%d, ", shared_cont_mat[i * k + j]);
            }
        }
        printf("\\n");
    }
}

"""

d_unravel_index_str = """
/**
 * @brief Unravel a flat index to the corresponding 2D indicis
 * @param[in] flat_idx The flat index to unravel
 * @param[in] num_cols Number of columns in the 2D array
 * @param[out] row Pointer to the row index
 * @param[out] col Pointer to the column index
 */
extern "C" __device__ __host__ inline void unravel_index(int flat_idx, int num_cols, int* row, int* col) {
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
 * @param parts 3D Array of partitions with shape of (n_features, n_parts, n_objs)
 * @param n_aris Number of ARIs to compute
 * @param k The max value of cluster number + 1
 * @param out Output array of ARIs
 * @param part_pairs Output array of part pairs to be compared by ARI
 */
extern "C" __global__ void ari(int *parts,
                    const int n_aris,
                    const int n_features,
                    const int n_parts,
                    const int n_objs,
                    const int n_elems_per_feat,
                    const int n_part_mat_elems,
                    const int k,
                    float *out,
                    int *part_pairs = nullptr)
{
    /*
     * Step 1: Each thead, unravel flat indices and load the corresponding data into shared memory
     */
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // each block is responsible for one ARI computation
    int ari_block_idx = blockIdx.x;

    // print parts for debugging


    // obtain the corresponding parts and unique counts
    // printf("n_part_mat_elems: %d\\n", n_part_mat_elems);
    int feature_comp_flat_idx = ari_block_idx / n_part_mat_elems; // flat comparison pair index for two features
    int part_pair_flat_idx = ari_block_idx % n_part_mat_elems;    // flat comparison pair index for two partitions of one feature pair
    int i, j;

    // if (global_tid == 0)
    // {
    //     printf("ari_block_idx: %d, feature_comp_flat_idx: %d, part_pair_flat_idx: %d\\n", ari_block_idx, feature_comp_flat_idx, part_pair_flat_idx);
    // }

    // unravel the feature indices
    get_coords_from_index(n_features, feature_comp_flat_idx, &i, &j);
    assert(i < n_features && j < n_features);
    assert(i >= 0 && j >= 0);
    // if (global_tid == 0)
    // {
    //     printf("global_tid: %d, i: %d, j: %d\\n", global_tid, i, j);
    // }
    // unravel the partition indices
    int m, n;
    unravel_index(part_pair_flat_idx, n_parts, &m, &n);
    // if (global_tid == 0)
    // {
    //     printf("global_tid: %d, m: %d, n: %d\\n", global_tid, m, n);
    // }

    // Make pointers to select the parts and unique counts for the feature pair
    // Todo: Use int4*?
    int *t_data_part0 = parts + i * n_elems_per_feat + m * n_objs; // t_ for thread
    int *t_data_part1 = parts + j * n_elems_per_feat + n * n_objs;

    // Load gmem data into smem by using different threads
    extern __shared__ int shared_mem[];
    int *s_part0 = shared_mem;
    int *s_part1 = shared_mem + n_objs;

    // Loop over the data using the block-stride pattern
    for (int i = threadIdx.x; i < n_objs; i += blockDim.x)
    {
        s_part0[i] = t_data_part0[i];
        s_part1[i] = t_data_part1[i];
    }
    __syncthreads();

    // Copy data to global memory if part_pairs is specified
    if (part_pairs != nullptr)
    {
        int *out_part0 = part_pairs + ari_block_idx * (2 * n_objs);
        int *out_part1 = out_part0 + n_objs;

        for (int i = threadIdx.x; i < n_objs; i += blockDim.x)
        {
            out_part0[i] = s_part0[i];
            out_part1[i] = s_part1[i];
        }
    }

    /*
     * Step 2: Compute contingency matrix within the block
     */
    // shared mem address for the contingency matrix
    int *s_contingency = shared_mem + 2 * n_objs;
    // initialize the contingency matrix to zero
    // const int n_contingency_items = k * k;
    // for (int i = threadIdx.x; i < n_contingency_items; i += blockDim.x) {
    //     s_contingency[i] = 0;
    // }
    get_contingency_matrix(t_data_part0, t_data_part1, n_objs, s_contingency, k);
    // if (global_tid == 0)
    // {
    //     for (int i = 0; i < k; ++i)
    //     {
    //         for (int j = 0; j < k; ++j)
    //         {
    //             printf("s_contingency[%d][%d]: %d\\n", i, j, s_contingency[i * k + j]);
    //         }
    //     }
    // }

    /*
     * Step 3: Construct pair confusion matrix
     */
    // shared mem address for the pair confusion matrix
    int *s_sum_rows = s_contingency + k * k;
    int *s_sum_cols = s_sum_rows + k;
    int *s_pair_confusion_matrix = s_sum_cols + k;
    get_pair_confusion_matrix(s_contingency, s_sum_rows, s_sum_cols, n_objs, k, s_pair_confusion_matrix);
    /*
     * Step 4: Compute ARI and write to global memory
     */
    if (threadIdx.x == 0) {
        int tn = static_cast<float>(s_pair_confusion_matrix[0]);
        int fp = static_cast<float>(s_pair_confusion_matrix[1]);
        int fn = static_cast<float>(s_pair_confusion_matrix[2]);
        int tp = static_cast<float>(s_pair_confusion_matrix[3]);
        printf("tn: %d, fp: %d, fn: %d, tp: %d\\n", tn, fp, fn, tp);
        float ari = 0.0;
        if (fn == 0 && fp == 0) {
            ari = 1.0;
        } else {
            ari = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
        }
        printf("ari: %f\\n", ari);
        out[ari_block_idx] = ari;
    }
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


def ari_dim2(feature_parts: cp.ndarray,
             n_partitions: int,
             n_features_comp: int,
             out: cp.ndarray,
             unique_element_counts: cp.ndarray):
    """
    Function to compute the ARI between partitions on the GPU. This function is responsible for launching the kernel
    in different streams for each pair of partitions.

    Args:
        feature_parts: 3D device array with cluster assignments for x features, y partitions, and z objects.
        Example initialization for this array: d_parts = cp.empty((nx, ny, nz), dtype=np.int16) - 1

        n_partitions: Number of partitions of a feature to compare.

        n_features_comp: Pre-computed number of features to compare.

        out: Pointer to the pre-allocated 1D device output array with length of n_features_comp.

        unique_element_counts: 2D device array with the number of unique elements for feature x and partition y.
    """

    # Can use non-blocking CPU scheduling or CUDA dynamic parallelism to launch the kernel for each pair of partitions.

    # Get metadata
    n_features, n_parts, n_objs = feature_parts.shape

    # Each kernel launch will be responsible for computing the ARI between two partitions.
    n_part_mat_elems = n_partitions * n_partitions
    # Each thread
    n_ari_pairs = n_partitions * n_part_mat_elems
    cm_values = cp.full(n_features_comp, cp.nan)
    # Todo: how many ari pairs? n_range_cluster?
    threads_per_block = 1
    blocks_per_grid = (n_ari_pairs + threads_per_block - 1) // threads_per_block

    ari_kernel = get_kernel()
    # Todo: use different streams?
    # Allocate output arrays for parts (debugging)
    out_parts0 = cp.empty(n_objs, dtype=np.int32)
    out_parts1 = cp.empty(n_objs, dtype=np.int32)
    shared_mem_size = 2 * n_objs

    # Launch the kernel, using one block per ARI
    ari_kernel(grid=(blocks_per_grid,),
               block=(threads_per_block,),
               shared_mem=shared_mem_size,
               args=(feature_parts, unique_element_counts, n_features_comp, n_part_mat_elems, out))

    raise NotImplementedError("Not implemented yet")
