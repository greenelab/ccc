#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>

#include <iostream>
#include <cmath>
#include <assert.h>
#include "metrics.cuh"

namespace py = pybind11;

/**
 * Future optimizations
 * 1. use narrower data types
 * 2. optimized on locality
 * 3. use warp-level reduction
 */


// Todo: Add CudaCheckError
#define gpuErrorCheck(ans, abort) \
{ \
    gpuAssert((ans), __FILE__, __LINE__, abort); \
}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}
// // call like this
// gpuErrorCheck(cudaMalloc(...)); // if fails, print message and continue
// gpuErrorCheck(cudaMalloc(...), true); // if fails, print message and abort

/**
 * @brief Unravel a flat index to the corresponding 2D indicis
 * @param[in] flat_idx The flat index to unravel
 * @param[in] num_cols Number of columns in the 2D array
 * @param[out] row Pointer to the row index
 * @param[out] col Pointer to the column index
 */
__device__ __host__ inline void unravel_index(int flat_idx, int num_cols, int *row, int *col)
{
    // change int to uint32_t
    *row = flat_idx / num_cols; // Compute row index
    *col = flat_idx % num_cols; // Compute column index
}

/**
 * @brief Given the number of objects and an index, this function calculates
 *        the coordinates in a symmetric matrix from a flat index.
 *        For example, if there are n_obj objects (such as genes), a condensed
 *        1D array can be created with pairwise comparisons between these
 *        objects, which corresponds to a symmetric 2D matrix. This function
 *        calculates the 2D coordinates (x, y) in the symmetric matrix that
 *        corresponds to the given flat index.
 *
 * @param[in] n_obj The total number of objects (i.e., the size of one dimension
 *                  of the square symmetric matrix).
 * @param[in] idx The flat index from the condensed pairwise array.
 * @param[out] x Pointer to the calculated row coordinate in the symmetric matrix.
 * @param[out] y Pointer to the calculated column coordinate in the symmetric matrix.
 */
__device__ __host__ inline void get_coords_from_index(int n_obj, int idx, int *x, int *y)
{
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

/**
 * @brief Compute the contingency matrix for two partitions using shared memory
 * @param[in] part0 Pointer to the first partition array
 * @param[in] part1 Pointer to the second partition array
 * @param[in] n Number of elements in each partition array
 * @param[out] shared_cont_mat Pointer to shared memory for storing the contingency matrix
 * @param[in] k Maximum number of clusters (size of contingency matrix is k x k)
 */
__device__ void get_contingency_matrix(int *part0, int *part1, int n, int *shared_cont_mat, int k)
{
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int size = k * k;

    // Initialize shared memory
    for (int i = tid; i < size; i += num_threads)
    {
        shared_cont_mat[i] = 0;
    }
    __syncthreads();

    // Process elements with bounds checking
    for (int i = tid; i < n; i += num_threads)
    {
        int row = part0[i];
        int col = part1[i];

        // Add bounds checking
        if (row >= 0 && row < k && col >= 0 && col < k)
        {
            atomicAdd(&shared_cont_mat[row * k + col], 1);
        }
    }
    __syncthreads();
}

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
    const int *__restrict__ contingency,
    int *sum_rows,
    int *sum_cols,
    const int n_objs,
    const int k,
    int *C)
{
    // Initialize sum_rows and sum_cols
    for (int i = threadIdx.x; i < k; i += blockDim.x)
    {
        sum_rows[i] = 0;
        sum_cols[i] = 0;
    }
    __syncthreads();

    // Compute sum_rows and sum_cols
    for (int i = threadIdx.x; i < k * k; i += blockDim.x)
    {
        int row = i / k;
        int col = i % k;
        int val = contingency[i];
        atomicAdd(&sum_cols[col], val);
        atomicAdd(&sum_rows[row], val);
    }
    __syncthreads();

    // Compute sum_squares
    int sum_squares;
    if (threadIdx.x == 0)
    {
        sum_squares = 0;
        for (int i = 0; i < k * k; ++i)
        {
            sum_squares += (contingency[i]) * contingency[i];
        }
    }
    __syncthreads();
    // printf("sum_squares: %d\n", sum_squares);

    // Compute C[1,1], C[0,1], C[1,0], and C[0,0]
    if (threadIdx.x == 0)
    {
        C[3] = sum_squares - n_objs; // C[1,1]

        int temp = 0;
        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                temp += (contingency[i * k + j]) * sum_cols[j];
            }
        }
        C[1] = temp - sum_squares; // C[0,1]

        temp = 0;
        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                temp += (contingency[j * k + i]) * sum_rows[j];
            }
        }
        C[2] = temp - sum_squares; // C[1,0]

        C[0] = n_objs * n_objs - C[1] - C[2] - sum_squares; // C[0,0]
    }
}

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
extern "C"
__global__ void ari(int *parts,
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
     * Step 0: Compute shared memory addresses
     */
    extern __shared__ int shared_mem[];
    int *s_part0 = shared_mem;                        // n_objs elements
    int *s_part1 = s_part0 + n_objs;                 // n_objs elements
    int *s_contingency = s_part1 + n_objs;           // k * k elements
    int *s_sum_rows = s_contingency + (k * k);       // k elements
    int *s_sum_cols = s_sum_rows + k;                // k elements
    int *s_pair_confusion_matrix = s_sum_cols + k;   // 4 elements

    /*
     * Step 1: Each thead, unravel flat indices and load the corresponding data into shared memory
     */
    // each block is responsible for one ARI computation
    int ari_block_idx = blockIdx.x;
    // obtain the corresponding parts and unique counts
    int feature_comp_flat_idx = ari_block_idx / n_part_mat_elems; // flat comparison pair index for two features
    int part_pair_flat_idx = ari_block_idx % n_part_mat_elems;    // flat comparison pair index for two partitions of one feature pair
    int i, j;
    // unravel the feature indices
    get_coords_from_index(n_features, feature_comp_flat_idx, &i, &j);
    assert(i < n_features && j < n_features);
    assert(i >= 0 && j >= 0);
    // unravel the partition indices
    int m, n;
    unravel_index(part_pair_flat_idx, n_parts, &m, &n);
    // Make pointers to select the parts and unique counts for the feature pair
    // Todo: Use int4*?
    int *t_data_part0 = parts + i * n_elems_per_feat + m * n_objs; // t_ for thread
    int *t_data_part1 = parts + j * n_elems_per_feat + n * n_objs;

    // Load gmem data into smem by using different threads
    // extern __shared__ int shared_mem[];
    // int *s_part0 = shared_mem;
    // int *s_part1 = shared_mem + n_objs;

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
    // int *s_contingency = shared_mem + 2 * n_objs;
    get_contingency_matrix(t_data_part0, t_data_part1, n_objs, s_contingency, k);

    /*
     * Step 3: Construct pair confusion matrix
     */
    // shared mem address for the pair confusion matrix
    // int *s_sum_rows = s_contingency + k * k;
    // int *s_sum_cols = s_sum_rows + k;
    // int *s_pair_confusion_matrix = s_sum_cols + k;
    get_pair_confusion_matrix(s_contingency, s_sum_rows, s_sum_cols, n_objs, k, s_pair_confusion_matrix);
    /*
     * Step 4: Compute ARI and write to global memory
     */
    if (threadIdx.x == 0)
    {
        int tn = static_cast<float>(s_pair_confusion_matrix[0]);
        int fp = static_cast<float>(s_pair_confusion_matrix[1]);
        int fn = static_cast<float>(s_pair_confusion_matrix[2]);
        int tp = static_cast<float>(s_pair_confusion_matrix[3]);
        float ari = 0.0;
        if (fn == 0 && fp == 0)
        {
            ari = 1.0;
        }
        else
        {
            ari = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
        }
        out[ari_block_idx] = ari;
    }
    __syncthreads();
}

/**
 * @brief Internal lower-level ARI computation
 * @param parts pointer to the 3D Array of partitions with shape of (n_features, n_parts, n_objs)
 * @throws std::invalid_argument if "parts" is invalid
 * @return std::vector<float> ARI values for each pair of partitions
 */
template <typename T>
auto ari_core(const T* parts, 
         const size_t n_features,
         const size_t n_parts,
         const size_t n_objs) -> std::vector<float> {
    /*
     * Notes for future bug fixing and optimization
     */
    // 1. GPU memory is not enough to store the partitions -> split the partitions into smaller chunks and do stream processing

    /*
     * Pre-computation
     */
    // Todo: dynamically query types
    using parts_dtype = T;
    using out_dtype = float;
    // Compute internal variables
    const auto n_feature_comp = n_features * (n_features - 1) / 2;
    const auto n_aris = n_feature_comp * n_parts * n_parts;

    /*
     * Memory Allocation
     */
    // Allocate host memory
    thrust::host_vector<out_dtype> h_out(n_aris);
    thrust::host_vector<parts_dtype> h_parts_pairs(n_aris * 2 * n_objs);
    // Allocate device memory with thrust
    // const int* parts_raw = parts[0][0].data();
    thrust::device_vector<parts_dtype> d_parts(parts, parts + n_features * n_parts * n_objs);   // data is copied to device
    thrust::device_vector<parts_dtype> d_parts_pairs(n_aris * 2 * n_objs);
    thrust::device_vector<out_dtype> d_out(n_aris);

    // Set up CUDA kernel configuration
    const auto block_size = 1024; // Todo: query device for max threads per block, older devices only support 512 threads per 1D block
    // Each block is responsible for one ARI computation
    const auto grid_size = n_aris;

    // Define shared memory size for each block
    // Compute k, the maximum value in d_parts + 1, used for shared memory allocation later
    const auto max_iter = thrust::max_element(d_parts.begin(), d_parts.end());
    const auto k = *max_iter + 1;
    const auto sz_parts_dtype = sizeof(parts_dtype);
    // FIXME: Compute shared memory size
    auto s_mem_size = 2 * n_objs * sz_parts_dtype;  // For the partition pair to be compared
    s_mem_size += k * k * sz_parts_dtype;           // For contingency matrix
    s_mem_size += 2 * n_parts * sz_parts_dtype;     // For the internal sum arrays
    s_mem_size += 4 * sz_parts_dtype;               // For the 2 x 2 confusion matrix

    /*
     * Launch the kernel
     */
    ari<<<grid_size, block_size, s_mem_size>>>(
        thrust::raw_pointer_cast(d_parts.data()),
        n_aris,
        n_features,
        n_parts,
        n_objs,
        n_parts * n_objs,
        n_parts * n_parts,
        k,
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_parts_pairs.data()));
    
    // Copy data back to host
    thrust::copy(d_out.begin(), d_out.end(), h_out.begin());
    thrust::copy(d_parts_pairs.begin(), d_parts_pairs.end(), h_parts_pairs.begin());

    // Free device memory

    // Convert thrust vectors to std::vector
    std::vector<float> res(h_out.begin(), h_out.end());

    // Return the ARI values
    return res;
}

/**
 * @brief API exposed to Python for computing ARI using CUDA upon a 3D Numpy NDArray of partitions
 * @param parts 3D Numpy.NDArray of partitions with shape of (n_features, n_parts, n_objs)
 * @throws std::invalid_argument if "parts" is invalid
 * @return std::vector<float> ARI values for each pair of partitions
 */
template <typename T>
auto ari(const py::array_t<T, py::array::c_style>& parts, 
             const size_t n_features,
             const size_t n_parts,
             const size_t n_objs) -> std::vector<float> {
    // Edge cases:
    // 1. GPU memory is not enough to store the partitions -> split the partitions into smaller chunks and do stream processing

    // Input processing
    // Request a buffer descriptor from Python
    py::buffer_info buffer = parts.request();

    // Some basic validation checks ...
    if (buffer.format != py::format_descriptor<T>::format())
        throw std::runtime_error("Incompatible format: expected an int array!");

    if (buffer.ndim != 3)
        throw std::runtime_error("Incompatible buffer dimension!");

    // Apply resources
    auto result = py::array_t<T>(buffer.size);

    // Obtain numpy.ndarray data pointer
    const auto parts_ptr = static_cast<T*>(buffer.ptr);

    return ari_core(parts_ptr, n_features, n_parts, n_objs);
}


// Below is the explicit instantiation of the ari template function.
//
// Generally people would write the implementation of template classes and functions in the header file. However, we
// separate the implementation into a .cpp file to make things clearer. In order to make the compiler know the
// implementation of the template functions, we need to explicitly instantiate them here, so that they can be picked up
// by the linker.

template auto ari<int>(const py::array_t<int, py::array::c_style>& parts, const size_t n_features, const size_t n_parts, const size_t n_objs) -> std::vector<float>;
template auto ari_core<int>(const int* parts, const size_t n_features, const size_t n_parts, const size_t n_objs) -> std::vector<float>;
