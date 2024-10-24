#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <assert.h>

// Todo: Add CudaCheckError

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
    int bid = blockIdx.x;
    int num_threads = blockDim.x;
    int num_blocks = gridDim.x;
    int size = k * k;

    // Initialize shared memory
    for (int i = tid; i < size; i += num_threads)
    {
        shared_cont_mat[i] = 0;
    }
    __syncthreads();

    // Process elements
    for (int i = tid; i < n; i += num_threads)
    {
        int row = part0[i];
        int col = part1[i];

        if (row < k && col < k)
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
    
    // Compute sum_squares
    int sum_squares;
    if (threadIdx.x == 0) {
        sum_squares = 0;
        for (int i = 0; i < k * k; ++i) {
            sum_squares += (contingency[i]) * contingency[i];
        }
    }
    __syncthreads();
    // printf("sum_squares: %d\n", sum_squares);

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
        printf("C[0,0]: %d, C[0,1]: %d, C[1,0]: %d, C[1,1]: %d\n", C[0], C[1], C[2], C[3]);

        // compute ARI
        int tn = static_cast<float>(C[0]);
        int fp = static_cast<float>(C[1]);
        int fn = static_cast<float>(C[2]);
        int tp = static_cast<float>(C[3]);
        printf("tn: %d, fp: %d, fn: %d, tp: %d\n", tn, fp, fn, tp);
        float ari = 0.0;
        if (fn == 0 && fp ==0) {
            ari = 1.0;
        } else {
            ari = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
        }
        printf("ari: %f\n", ari);
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
     * Step 1: Each thead, unravel flat indices and load the corresponding data into shared memory
     */
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
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
    get_contingency_matrix(t_data_part0, t_data_part1, n_objs, s_contingency, k);

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
        printf("tn: %d, fp: %d, fn: %d, tp: %d\n", tn, fp, fn, tp);
        float ari = 0.0;
        if (fn == 0 && fp == 0) {
            ari = 1.0;
        } else {
            ari = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
        }
        printf("ari: %f\n", ari);
        out[ari_block_idx] = ari;
    }
    __syncthreads();

}

// Helper function to generate pairwise combinations (implement this according to your needs)
std::vector<std::pair<std::vector<int>, std::vector<int>>> generate_pairwise_combinations(const std::vector<std::vector<std::vector<int>>> &arr)
{
    std::vector<std::pair<std::vector<int>, std::vector<int>>> pairs;
    size_t num_slices = arr.size(); // Number of 2D arrays in the 3D vector
    for (size_t i = 0; i < num_slices; ++i)
    {
        for (size_t j = i + 1; j < num_slices; ++j)
        { // Only consider pairs in different slices
            for (const auto &row_i : arr[i])
            { // Each row in slice i
                for (const auto &row_j : arr[j])
                { // Pairs with each row in slice j
                    pairs.emplace_back(row_i, row_j);
                }
            }
        }
    }
    return pairs;
}

void test_ari_parts_selection()
{
    // Define test input
    std::vector<std::vector<std::vector<int>>> parts = {
        {{0, 1, 2, 3},
         {0, 2, 3, 4},
         {0, 3, 4, 5}},
        {{1, 1, 2, 3},
         {1, 2, 3, 4},
         {1, 3, 4, 5}},
        {{2, 1, 2, 3},
         {2, 2, 3, 4},
         {2, 3, 4, 5}}};

    const int k = 6; // specified by the call to ccc , part number from [0...9]
    
    // std::vector<std::vector<std::vector<int>>> parts = {
    //     {{4, 1, 3, 5, 2, 0, 6, 3, 1, 4},
    //     {0, 2, 6, 4, 5, 3, 1, 0, 6, 2},
    //     {1, 5, 3, 2, 4, 0, 6, 1, 5, 3}},
        
    //     // {{3, 6, 0, 2, 1, 5, 4, 3, 6, 0},
    //     // {5, 1, 4, 0, 3, 6, 2, 1, 5, 4},
    //     // {2, 3, 6, 1, 0, 5, 4, 3, 6, 2}},
        
    //     {{1, 4, 5, 3, 6, 0, 2, 5, 4, 1},
    //     {0, 6, 2, 5, 1, 3, 4, 6, 0, 2},
    //     {4, 1, 3, 6, 5, 0, 2, 4, 1, 3}}
    // };

    // const int k = 7; // specified by the call to ccc , max(parts) + 1
    

    // std::vector<int> part_maxes = {3, 4, 5, 3, 4, 5, 3, 4, 5};
    // auto sz_part_maxes = sizeof(part_maxes) / sizeof(part_maxes[0]);

    // Get dimensions
    int n_features = parts.size();
    int n_parts = parts[0].size();
    int n_objs = parts[0][0].size();
    int n_feature_comp = n_features * (n_features - 1) / 2;
    int n_aris = n_feature_comp * n_parts * n_parts;
    std::cout << "n_features: " << n_features << ", n_parts: " << n_parts << ", n_objs: " << n_objs << std::endl
              << "n_feature_comps: " << n_feature_comp << ", n_aris: " << n_aris << std::endl;

    // Allocate host memory for C-style array
    int *h_parts = new int[n_features * n_parts * n_objs];

    // Copy data from vector to C-style array
    for (int i = 0; i < n_features; ++i)
    {
        for (int j = 0; j < n_parts; ++j)
        {
            for (int k = 0; k < n_objs; ++k)
            {
                h_parts[i * (n_parts * n_objs) + j * n_objs + k] = parts[i][j][k];
            }
        }
    }

    // Set up CUDA kernel configuration
    int block_size = 2;
    // Each block is responsible for one ARI computation
    int grid_size = n_aris;
    // Compute shared memory size
    size_t s_mem_size = n_objs * 2 * sizeof(int); // For the partition pair to be compared
    s_mem_size += 2 * k * sizeof(int); // For the internal sum arrays
    s_mem_size += 4 * sizeof(int); // For the 2 x 2 confusion matrix

    // Allocate device memory
    int *d_parts, *d_parts_pairs;
    float *d_out;
    cudaMalloc(&d_parts, n_features * n_parts * n_objs * sizeof(int));
    cudaMalloc(&d_out, n_aris * sizeof(float));
    cudaMalloc(&d_parts_pairs, n_aris * 2 * n_objs * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_parts, h_parts, n_features * n_parts * n_objs * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    ari<<<grid_size, block_size, s_mem_size>>>(
        d_parts,
        n_aris,
        n_features,
        n_parts,
        n_objs,
        n_parts * n_objs,
        n_parts * n_parts,
        k,
        d_out,
        d_parts_pairs);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy results back to host
    int *h_parts_pairs = new int[n_aris * 2 * n_objs];
    cudaMemcpy(h_parts_pairs, d_parts_pairs, n_aris * 2 * n_objs * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Parts pairs: " << std::endl;
    for (int i = 0; i < n_aris; ++i)
    {
        std::cout << "Pair:" << i << std::endl;
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < n_objs; ++k)
            {
                std::cout << *(h_parts_pairs + i * 2 * n_objs + j * n_objs + k) << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Assert equality on the parts pairs
    bool all_equal = true;
    auto pairs = generate_pairwise_combinations(parts);
    int n_pairs = pairs.size();
    for (int i = 0; i < n_pairs; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            const std::vector<int> &current_vector = (j == 0) ? pairs[i].first : pairs[i].second;
            for (int k = 0; k < n_objs; ++k)
            {
                int flattened_index = i * 2 * n_objs + j * n_objs + k;
                if (h_parts_pairs[flattened_index] != current_vector[k])
                {
                    all_equal = false;
                    std::cout << "Mismatch at i=" << i << ", j=" << j << ", k=" << k << std::endl;
                    std::cout << "Expected: " << current_vector[k] << ", Got: " << h_parts_pairs[flattened_index] << std::endl;
                }
            }
        }
    }

    if (all_equal)
    {
        std::cout << "Test passed: All elements match." << std::endl;
    }
    else
    {
        std::cout << "Test failed: Mismatches found." << std::endl;
    }

    // Print ARI results
    float *h_out = new float[n_aris];
    cudaMemcpy(h_out, d_out, n_aris * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "ARI results: " << std::endl;
    for (int i = 0; i < n_aris; ++i)
    {
        printf("%f, ", h_out[i]);
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_parts);
    cudaFree(d_out);
    cudaFree(d_parts_pairs);
    delete[] h_parts_pairs;
}

int main()
{
    test_ari_parts_selection();
    return 0;
}