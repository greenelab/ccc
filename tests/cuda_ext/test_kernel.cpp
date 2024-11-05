#include <vector>
#include <gtest/gtest.h>
#include <pybind11/numpy.h>
#include "../../libs/ccc_cuda_ext/metrics.cuh"

namespace py = pybind11;

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


using Mat3 = std::vector<std::vector<std::vector<int>>>;
using TestParamType = std::tuple<Mat3, float>;

// Define a parameterized test fixture
class CudaAriTest : public ::testing::TestWithParam<TestParamType> {};

TEST_P(CudaAriTest, CheckSingleResult)
{
    Mat3 parts;
    float expected_result;
    std::tie(parts, expected_result) = GetParam();

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

    auto h_out = ari_core<int>(h_parts, n_features, n_parts, n_objs)[0];

    // Check if the result are close
    EXPECT_NEAR(h_out, expected_result, 1e-2);
}

// Instantiate the test suite with parameter values
// These tests are taken from sklearn.metrics.adjusted_rand_score:
// https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
INSTANTIATE_TEST_SUITE_P(
    CudaAriTestInstances,
    CudaAriTest,
    ::testing::Values(
        TestParamType(
            Mat3{
                {{0, 0, 1, 2}},
                {{0, 0, 1, 1}},
            },
            0.57f
        ),
        TestParamType(
            Mat3{
                {{0, 0, 1, 1}},
                {{0, 1, 0, 1}},
            },
            -0.5f
        ),
        TestParamType(
            Mat3{
                {{0, 0, 1, 1}},
                {{0, 0, 1, 1}},
            },
            1.0f
        ),
        TestParamType(
            Mat3{
                {{0, 0, 1, 1}},
                {{1, 1, 0, 0}},
            },
            1.0f
        ),
        TestParamType(
            Mat3{
                {{0, 0, 0, 0}},
                {{0, 1, 2, 3}},
            },
            0.0f
        )
    )
);
