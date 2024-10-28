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

    auto h_out = cudaAri(h_parts, n_features, n_parts, n_objs);

    // Print ARI results
    std::cout << "ARI results: " << std::endl;
    for (int i = 0; i < n_aris; ++i)
    {
        printf("%f, ", h_out[i]);
    }
    std::cout << std::endl;
}

int main()
{
    std::cout << "Hello, World!" << std::endl;
    test_ari_parts_selection();
    return 0;
}
