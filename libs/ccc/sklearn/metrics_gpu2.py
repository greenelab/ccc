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
#include <stdio.h>
#define debug 0

/**
 * @brief Main ARI kernel. Now only compare a pair of ARIs
 * @param n_parts Number of partitions of each feature
 * @param n_objs Number of objects in each partitions
 * @param n_part_mat_elems Number of elements in the square partition matrix
 * @param n_elems_per_feat Number of elements for each feature, i.e., part[i].x * part[i].y
 * @param parts 3D Array of partitions with shape of (n_features, n_parts, n_objs)
 * @param uniqs Array of unique counts
 * @param n_aris Number of ARIs to compute
 * @param out Output array of ARIs
 * @param part0 Output array of partition 0, for debugging
 * @param part1 Output array of partition 1, for debugging
 */
extern "C" __global__
void ari(int* parts,
         int* uniqs,
         const int n_aris,
         const int n_features,
         const int n_parts,
         const int n_objs,
         const int n_elems_per_feat,
         const int n_part_mat_elems,
         float* out,
         int* part_pairs
         )
{
    // tid is the block-wide thread index [0, blockDim.x]
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // each block is responsible for one ARI computation
    // int ari_block_idx = blockIdx.x;
    int ari_block_idx = 3;

    // obtain the corresponding parts and unique counts
    int feature_comp_flat_idx = ari_block_idx / n_part_mat_elems;   // flat comparison pair index for two features
    int part_pair_flat_idx = ari_block_idx % n_part_mat_elems;  // flat comparison pair index for two partitions of one feature pair
    int i, j;

    // print parts for debugging
    if (global_tid == 0) {
        for (int i = 0; i < n_features; ++i) {
            for (int j = 0; j < n_parts; ++j) {
                for (int k = 0; k < n_objs; ++k) {
                    printf("parts[%d][%d][%d]: %d\\n", i, j, k, parts[i * n_parts * n_objs + j * n_objs + k]);
                }
            }
            printf("\\n");
        }
    }

    // unravel the feature indices
    get_coords_from_index(n_parts, feature_comp_flat_idx, &i, &j);
    if (global_tid == 0) {
        printf("global_tid: %d, i: %d, j: %d\\n", global_tid, i, j);
    }
    // unravel the partition indices
    int m, n;
    unravel_index(part_pair_flat_idx, n_parts, &m, &n);
    if (global_tid == 0){
        printf("global_tid: %d, m: %d, n: %d\\n", global_tid, m, n);
    }
    
    // Make pointers to select the parts and unique counts for the feature pair
    // Todo: Use int4*?
    int* t_data_parti = parts + i * n_elems_per_feat + m * n_objs ;  // t_ for thread
    int* t_data_partj = parts + j * n_elems_per_feat + n * n_objs ;
    //int* t_data_uniqi = uniqs + i * n_parts + m;
    //int* t_data_uniqj = uniqs + j * n_parts + n;
    int* blk_part_pairs = part_pairs + ari_block_idx * (2 * n_objs);
    
    // Load gmem data into smem by using different threads
    extern __shared__ int shared_mem[];
    
    // Number of chunks of data this block will load
    // In case block size is smaller than the partition size
    const int num_chunks = (n_objs + blockDim.x - 1) / blockDim.x;
    // Loop over the chunks of data
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        // idx is the linear global memory index of the element to load
        int idx = chunk * blockDim.x + global_tid;

        if (idx < n_objs) {
            // Load part_i and part_j into shared memory
            shared_mem[tid] = t_data_parti[idx];
            shared_mem[tid + n_objs] = t_data_partj[idx];
            __syncthreads();  // Synchronize to ensure all threads have loaded data into shared memory

            // Each thread writes data back to global memory (for demonstration purposes)
            // part0[idx] = shared_mem[tid];
            // part1[idx] = shared_mem[tid + n_objs];
            blk_part_pairs[idx] = shared_mem[tid];
            blk_part_pairs[idx + n_objs] = shared_mem[tid + n_objs];
            __syncthreads();  // Synchronize before moving to the next chunk
        }
    }
        
    // Initialize shared memory
    // int part_mat_first_tid = tid * part_part_elems;
    
    // Todo: use a for loop to compute the ARI and do the max reduction
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
