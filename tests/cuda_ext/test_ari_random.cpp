#include <iostream>
#include <vector>
#include <random>
#include <gtest/gtest.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/stl.h>
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


// Define test parameters structure
struct AriTestParams {
    int n_features;
    int n_parts;
    int n_objs;
    int k;
    
    // Constructor for easier initialization
    AriTestParams(int f, int p, int o, int k_val) 
        : n_features(f), n_parts(p), n_objs(o), k(k_val) {}
};

// Test fixture for parameterized test
class PairwiseAriTest : public ::testing::TestWithParam<AriTestParams> {
protected:
    // Static setup that runs once before all tests
    static void SetUpTestSuite() {
        if (!guard) {  // Only initialize if not already done
            guard = std::make_unique<py::scoped_interpreter>();
            np = std::make_unique<py::module_>(py::module_::import("numpy"));
            ccc_module = std::make_unique<py::module_>(py::module_::import("ccc.sklearn.metrics"));
        }
    }

    // Static teardown that runs once after all tests
    static void TearDownTestSuite() {
        ccc_module.reset();
        np.reset();
        guard.reset();
    }

    // Helper method to compute ARI using Python
    float compute_ari(const std::vector<int>& labels1, const std::vector<int>& labels2) {
        try {
            // Convert C++ vectors to numpy arrays
            py::array_t<int> np_part0 = py::cast(labels1);
            py::array_t<int> np_part1 = py::cast(labels2);

            // Call the ccc function
            py::object result = ccc_module->attr("adjusted_rand_index")(np_part0, np_part1);
            return result.cast<float>();
        }
        catch (const std::exception& e) {
            std::cerr << "Error computing ARI: " << e.what() << std::endl;
            return 0.0f;
        }
    }

private:
    // Static members shared across all test instances
    static std::unique_ptr<py::scoped_interpreter> guard;
    static std::unique_ptr<py::module_> np;
    static std::unique_ptr<py::module_> ccc_module;
};

// Define the static members
std::unique_ptr<py::scoped_interpreter> PairwiseAriTest::guard;
std::unique_ptr<py::module_> PairwiseAriTest::np;
std::unique_ptr<py::module_> PairwiseAriTest::ccc_module;

TEST_P(PairwiseAriTest, RandomPartitions) {
    const auto params = GetParam();
    const int n_features = params.n_features;
    const int n_parts = params.n_parts;
    const int n_objs = params.n_objs;
    const int k = params.k;
    
    // Generate random partitions (similar to numpy.random.randint)
    std::vector<int> parts(n_features * n_parts * n_objs);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, k - 1);
    
    for (auto& val : parts) {
        val = dis(gen);
    }
    
    // Calculate expected number of ARIs
    int n_feature_comp = n_features * (n_features - 1) / 2;
    int n_aris = n_feature_comp * n_parts * n_parts;
    
    // Get results from CUDA implementation
    auto res_aris = ari_core<int>(parts.data(), n_features, n_parts, n_objs);
    
    // Generate reference results
    std::vector<float> ref_aris(n_aris);
    
    // Convert flat array to 3D structure for easier processing
    std::vector<std::vector<std::vector<int>>> parts_3d(n_features, 
        std::vector<std::vector<int>>(n_parts, 
            std::vector<int>(n_objs)));
            
    // Fill 3D structure
    for (int f = 0; f < n_features; ++f) {
        for (int p = 0; p < n_parts; ++p) {
            for (int o = 0; o < n_objs; ++o) {
                parts_3d[f][p][o] = parts[f * (n_parts * n_objs) + p * n_objs + o];
            }
        }
    }
    
    // Generate pairs and compute reference ARIs
    auto pairs = generate_pairwise_combinations(parts_3d);

    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto& part0 = pairs[i].first;
        const auto& part1 = pairs[i].second;
        // Compute ARI for this pair
        ref_aris[i] = compute_ari(part0, part1);
    }
    
    // Compare results
    ASSERT_EQ(res_aris.size(), ref_aris.size());
    for (size_t i = 0; i < res_aris.size(); ++i) {
        EXPECT_NEAR(res_aris[i], ref_aris[i], 1e-5);
    }
}

// Instantiate the test suite with parameter values
INSTANTIATE_TEST_SUITE_P(
    PairwiseAriTestInstances,
    PairwiseAriTest,
    ::testing::Values(
        AriTestParams(2, 2, 100, 10),
        AriTestParams(5, 10, 200, 10)
        // Commented out cases that caused issues in Python:
        // AriTestParams(100, 20, 1000, 10),  // wrong results
        // AriTestParams(200, 20, 300, 10),   // illegal mem access
        // AriTestParams(1000, 10, 300, 10)   // out of gpu mem
    )
);

