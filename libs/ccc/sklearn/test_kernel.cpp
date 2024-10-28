#include <vector>
#include <iostream>
#include "metrics.cuh"

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

// void test_ari_parts_selection()
// {
//     // Define test input
//     std::vector<std::vector<std::vector<int>>> parts = {
//         {{0, 1, 2, 3},
//          {0, 2, 3, 4},
//          {0, 3, 4, 5}},
//         {{1, 1, 2, 3},
//          {1, 2, 3, 4},
//          {1, 3, 4, 5}},
//         {{2, 1, 2, 3},
//          {2, 2, 3, 4},
//          {2, 3, 4, 5}}};

//     const int k = 6; // specified by the call to ccc , part number from [0...9]

//     // std::vector<std::vector<std::vector<int>>> parts = {
//     //     {{4, 1, 3, 5, 2, 0, 6, 3, 1, 4},
//     //     {0, 2, 6, 4, 5, 3, 1, 0, 6, 2},
//     //     {1, 5, 3, 2, 4, 0, 6, 1, 5, 3}},

//     //     // {{3, 6, 0, 2, 1, 5, 4, 3, 6, 0},
//     //     // {5, 1, 4, 0, 3, 6, 2, 1, 5, 4},
//     //     // {2, 3, 6, 1, 0, 5, 4, 3, 6, 2}},

//     //     {{1, 4, 5, 3, 6, 0, 2, 5, 4, 1},
//     //     {0, 6, 2, 5, 1, 3, 4, 6, 0, 2},
//     //     {4, 1, 3, 6, 5, 0, 2, 4, 1, 3}}
//     // };

//     // const int k = 7; // specified by the call to ccc , max(parts) + 1

//     // std::vector<int> part_maxes = {3, 4, 5, 3, 4, 5, 3, 4, 5};
//     // auto sz_part_maxes = sizeof(part_maxes) / sizeof(part_maxes[0]);

//     // Get dimensions
//     int n_features = parts.size();
//     int n_parts = parts[0].size();
//     int n_objs = parts[0][0].size();
//     int n_feature_comp = n_features * (n_features - 1) / 2;
//     int n_aris = n_feature_comp * n_parts * n_parts;
//     std::cout << "n_features: " << n_features << ", n_parts: " << n_parts << ", n_objs: " << n_objs << std::endl
//               << "n_feature_comps: " << n_feature_comp << ", n_aris: " << n_aris << std::endl;

//     // Allocate host memory for C-style array
//     int *h_parts = new int[n_features * n_parts * n_objs];

//     // Copy data from vector to C-style array
//     for (int i = 0; i < n_features; ++i)
//     {
//         for (int j = 0; j < n_parts; ++j)
//         {
//             for (int k = 0; k < n_objs; ++k)
//             {
//                 h_parts[i * (n_parts * n_objs) + j * n_objs + k] = parts[i][j][k];
//             }
//         }
//     }

//     // Set up CUDA kernel configuration
//     int block_size = 2;
//     // Each block is responsible for one ARI computation
//     int grid_size = n_aris;
//     // Compute shared memory size
//     size_t s_mem_size = n_objs * 2 * sizeof(int); // For the partition pair to be compared
//     s_mem_size += 2 * k * sizeof(int);            // For the internal sum arrays
//     s_mem_size += 4 * sizeof(int);                // For the 2 x 2 confusion matrix

//     // Allocate device memory
//     int *d_parts, *d_parts_pairs;
//     float *d_out;
//     cudaMalloc(&d_parts, n_features * n_parts * n_objs * sizeof(int));
//     cudaMalloc(&d_out, n_aris * sizeof(float));
//     cudaMalloc(&d_parts_pairs, n_aris * 2 * n_objs * sizeof(int));

//     // Copy data to device
//     cudaMemcpy(d_parts, h_parts, n_features * n_parts * n_objs * sizeof(int), cudaMemcpyHostToDevice);

//     // Launch kernel
//     ari<<<grid_size, block_size, s_mem_size>>>(
//         d_parts,
//         n_aris,
//         n_features,
//         n_parts,
//         n_objs,
//         n_parts * n_objs,
//         n_parts * n_parts,
//         k,
//         d_out,
//         d_parts_pairs);

//     // Synchronize device
//     cudaDeviceSynchronize();

//     // Copy results back to host
//     int *h_parts_pairs = new int[n_aris * 2 * n_objs];
//     cudaMemcpy(h_parts_pairs, d_parts_pairs, n_aris * 2 * n_objs * sizeof(int), cudaMemcpyDeviceToHost);

//     // Print results
//     std::cout << "Parts pairs: " << std::endl;
//     for (int i = 0; i < n_aris; ++i)
//     {
//         std::cout << "Pair:" << i << std::endl;
//         for (int j = 0; j < 2; ++j)
//         {
//             for (int k = 0; k < n_objs; ++k)
//             {
//                 std::cout << *(h_parts_pairs + i * 2 * n_objs + j * n_objs + k) << ", ";
//             }
//             std::cout << std::endl;
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;

//     // Assert equality on the parts pairs
//     bool all_equal = true;
//     auto pairs = generate_pairwise_combinations(parts);
//     int n_pairs = pairs.size();
//     for (int i = 0; i < n_pairs; ++i)
//     {
//         for (int j = 0; j < 2; ++j)
//         {
//             const std::vector<int> &current_vector = (j == 0) ? pairs[i].first : pairs[i].second;
//             for (int k = 0; k < n_objs; ++k)
//             {
//                 int flattened_index = i * 2 * n_objs + j * n_objs + k;
//                 if (h_parts_pairs[flattened_index] != current_vector[k])
//                 {
//                     all_equal = false;
//                     std::cout << "Mismatch at i=" << i << ", j=" << j << ", k=" << k << std::endl;
//                     std::cout << "Expected: " << current_vector[k] << ", Got: " << h_parts_pairs[flattened_index] << std::endl;
//                 }
//             }
//         }
//     }

//     if (all_equal)
//     {
//         std::cout << "Test passed: All elements match." << std::endl;
//     }
//     else
//     {
//         std::cout << "Test failed: Mismatches found." << std::endl;
//     }

//     // Print ARI results
//     float *h_out = new float[n_aris];
//     cudaMemcpy(h_out, d_out, n_aris * sizeof(float), cudaMemcpyDeviceToHost);
//     std::cout << "ARI results: " << std::endl;
//     for (int i = 0; i < n_aris; ++i)
//     {
//         printf("%f, ", h_out[i]);
//     }
//     std::cout << std::endl;

//     // Clean up
//     cudaFree(d_parts);
//     cudaFree(d_out);
//     cudaFree(d_parts_pairs);
//     delete[] h_parts_pairs;
// }

int main()
{
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
