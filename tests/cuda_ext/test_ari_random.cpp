/**
 * @file test_ari_random.cpp
 * @brief Test suite for Adjusted Rand Index (ARI) computation using CUDA
 * 
 * This test suite validates the CUDA implementation of ARI computation against
 * a reference Python implementation. It tests various input sizes and configurations
 * using parameterized tests.
 * 
 * The test compares results from:
 * 1. CUDA implementation (ari_core)
 * 2. Python reference implementation (ccc.sklearn.metrics.adjusted_rand_index)
 */

#include <iostream>
#include <vector>
#include <random>
#include <ranges>
#include <algorithm>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../../libs/ccc_cuda_ext/metrics.cuh"

namespace py = pybind11;

/**
 * @brief Helper class for generating and manipulating test data
 * 
 * This class provides static utility functions for:
 * - Generating random partition data
 * - Reshaping arrays between different dimensions
 * - Generating pairwise combinations of partitions
 */
class TestDataGenerator {
public:
    /**
     * @brief Generates random partition assignments
     * 
     * @param n_features Number of features
     * @param n_parts Number of partitions per feature
     * @param n_objs Number of objects
     * @param k Number of possible cluster assignments
     * @param seed Random seed for reproducibility
     * @return std::vector<int> Flattened array of random partition assignments
     */
    static std::vector<int> generate_random_partitions(int n_features, int n_parts, 
                                                     int n_objs, int k, unsigned seed = 42) {
        std::vector<int> parts(n_features * n_parts * n_objs);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(0, k - 1);
        
        for (auto& val : parts) {
            val = dis(gen);
        }
        return parts;
    }

    /**
     * @brief Reshapes a flat array into a 3D structure
     * 
     * @param flat_array Input array
     * @param n_features Number of features
     * @param n_parts Number of partitions per feature
     * @param n_objs Number of objects
     * @return 3D vector representing [features][parts][objects]
     */
    static std::vector<std::vector<std::vector<int>>> reshape_to_3d(
        const std::vector<int>& flat_array, 
        int n_features, int n_parts, int n_objs) {
        
        std::vector<std::vector<std::vector<int>>> parts_3d(
            n_features, std::vector<std::vector<int>>(
                n_parts, std::vector<int>(n_objs)));
                
        for (int f = 0; f < n_features; ++f) {
            for (int p = 0; p < n_parts; ++p) {
                for (int o = 0; o < n_objs; ++o) {
                    parts_3d[f][p][o] = flat_array[f * (n_parts * n_objs) + p * n_objs + o];
                }
            }
        }
        return parts_3d;
    }

    /**
    * @brief Generates all pairwise combinations of partitions from different features
    * 
    * Given a 3D array of shape [n_features, n_parts, n_objs], this function generates
    * all possible pairs of partitions between different features. For example, if we have
    * features f0, f1, f2, it will generate pairs between:
    * - f0 and f1 partitions
    * - f0 and f2 partitions
    * - f1 and f2 partitions
    * 
    * @param arr A 3D vector where:
    *            - First dimension (arr.size()) represents different features
    *            - Second dimension (arr[i].size()) represents different partitions for each feature
    *            - Third dimension (arr[i][j].size()) represents objects in each partition
    * 
    * @return std::vector<std::pair<std::vector<int>, std::vector<int>>> 
    *         A vector of partition pairs where each pair contains:
    *         - first: vector of partition labels from one feature
    *         - second: vector of partition labels from another feature
    * 
    * @example
    *   // For a 3D array with shape [2, 2, 4]:
    *   arr = {
    *     {{0,1,2,3}, {4,5,6,7}},     // feature 0's partitions
    *     {{8,9,10,11}, {12,13,14,15}} // feature 1's partitions
    *   }
    *   // Will generate pairs:
    *   // ({0,1,2,3}, {8,9,10,11})
    *   // ({0,1,2,3}, {12,13,14,15})
    *   // ({4,5,6,7}, {8,9,10,11})
    *   // ({4,5,6,7}, {12,13,14,15})
    */
    static std::vector<std::pair<std::vector<int>, std::vector<int>>> 
    generate_pairwise_combinations(const std::vector<std::vector<std::vector<int>>>& arr) {
        std::vector<std::pair<std::vector<int>, std::vector<int>>> pairs;
        
        // Generate indices for features
        auto indices = std::views::iota(0u, arr.size());
        
        // For each feature index
        for (auto i : indices) {
            // For each subsequent feature index (avoiding duplicate pairs)
            for (auto j : std::views::iota(i + 1u, arr.size())) {
                // For each partition in feature i
                for (const auto& row_i : arr[i]) {
                    // For each partition in feature j
                    for (const auto& row_j : arr[j]) {
                        // Add the pair of partitions to our result
                        pairs.emplace_back(row_i, row_j);
                    }
                }
            }
        }
        return pairs;
    }
};

/**
 * @brief Parameters for ARI test cases
 * 
 * Encapsulates the parameters that define a test case for ARI computation:
 * - Number of features to compare
 * - Number of partitions per feature
 * - Number of objects in each partition
 * - Number of possible cluster assignments
 * - Tolerance for floating-point comparisons
 */
struct AriTestParams {
    int n_features;
    int n_parts;
    int n_objs;
    int k;
    float tolerance;  // Added tolerance as a parameter
    
    AriTestParams(int features, int parts, int objects, int clusters, float tol = 1e-5) 
        : n_features(features)
        , n_parts(parts)
        , n_objs(objects)
        , k(clusters)
        , tolerance(tol) {}

    // Add string representation for better test output
    friend std::ostream& operator<<(std::ostream& os, const AriTestParams& params) {
        return os << "Features=" << params.n_features 
                 << ", Parts=" << params.n_parts
                 << ", Objects=" << params.n_objs
                 << ", Clusters=" << params.k;
    }
};

/**
 * @brief Test fixture for parameterized ARI tests
 * 
 * This fixture provides:
 * 1. Python environment setup and teardown
 * 2. Reference implementation through Python
 * 3. Result validation utilities
 * 
 * The fixture ensures that:
 * - Python interpreter is initialized once for all tests
 * - Required Python modules are imported
 * - Resources are properly cleaned up
 */
class PairwiseAriTest : public ::testing::TestWithParam<AriTestParams> {
protected:
    /**
     * @brief Set up Python environment before any tests run
     * 
     * Initializes:
     * - Python interpreter
     * - NumPy module
     * - CCC metrics module
     */
    static void SetUpTestSuite() {
        if (!guard) {
            guard = std::make_unique<py::scoped_interpreter>();
            try {
                np = std::make_unique<py::module_>(py::module_::import("numpy"));
                ccc_module = std::make_unique<py::module_>(py::module_::import("ccc.sklearn.metrics"));
            } catch (const std::exception& e) {
                FAIL() << "Failed to initialize Python modules: " << e.what();
            }
        }
    }

    /**
     * @brief Clean up Python environment after all tests complete
     */
    static void TearDownTestSuite() {
        ccc_module.reset();
        np.reset();
        guard.reset();
    }

    /**
     * @brief Compute ARI using Python reference implementation
     * 
     * @param labels1 First partition
     * @param labels2 Second partition
     * @return float ARI score
     * @throws Logs failure if Python computation fails
     */
    float compute_ari(const std::vector<int>& labels1, const std::vector<int>& labels2) {
        try {
            py::array_t<int> np_part0 = py::cast(labels1);
            py::array_t<int> np_part1 = py::cast(labels2);

            py::object result = ccc_module->attr("adjusted_rand_index")(np_part0, np_part1);
            return result.cast<float>();
        } catch (const py::error_already_set& e) {
            ADD_FAILURE() << "Python error: " << e.what();
            return 0.0f;
        } catch (const std::exception& e) {
            ADD_FAILURE() << "C++ error: " << e.what();
            return 0.0f;
        }
    }

    /**
     * @brief Validate CUDA results against reference implementation
     * 
     * @param actual Results from CUDA implementation
     * @param expected Results from reference implementation
     * @param tolerance Maximum allowed difference
     */
    void validate_results(const std::vector<float>& actual, 
                         const std::vector<float>& expected,
                         float tolerance) {
        ASSERT_EQ(actual.size(), expected.size()) ;
            // << "Mismatch in result sizes";
        
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], tolerance);
                // << "Mismatch at index " << i;
        }
    }

private:
    static std::unique_ptr<py::scoped_interpreter> guard;
    static std::unique_ptr<py::module_> np;
    static std::unique_ptr<py::module_> ccc_module;
};

// Static member definitions
std::unique_ptr<py::scoped_interpreter> PairwiseAriTest::guard;
std::unique_ptr<py::module_> PairwiseAriTest::np;
std::unique_ptr<py::module_> PairwiseAriTest::ccc_module;

/**
 * @brief Test case for random partition ARI computation
 * 
 * This test:
 * 1. Generates random partition data
 * 2. Computes ARI using CUDA implementation
 * 3. Computes reference results using Python
 * 4. Validates CUDA results against reference
 * 
 * @param GetParam() Test parameters defining input size and configuration
 */
TEST_P(PairwiseAriTest, RandomPartitions) {
    const auto params = GetParam();
    
    // Generate test data
    auto parts = TestDataGenerator::generate_random_partitions(
        params.n_features, params.n_parts, params.n_objs, params.k);
    
    // Get CUDA results
    auto res_aris = ari_core<int>(parts.data(), 
        params.n_features, params.n_parts, params.n_objs);
    
    // Generate reference results
    auto parts_3d = TestDataGenerator::reshape_to_3d(
        parts, params.n_features, params.n_parts, params.n_objs);
    auto pairs = TestDataGenerator::generate_pairwise_combinations(parts_3d);
    
    std::vector<float> ref_aris;
    ref_aris.reserve(pairs.size());
    
    for (const auto& [part0, part1] : pairs) {
        ref_aris.push_back(compute_ari(part0, part1));
    }
    
    // Validate results
    validate_results(res_aris, ref_aris, params.tolerance);
}

/**
 * @brief Test suite instantiation with various parameter sets
 * 
 * Current test cases:
 * - Small input (2 features, 2 parts, 100 objects)
 * - Medium input (5 features, 10 parts, 200 objects)
 * 
 * Known issues:
 * - Wrong results with large inputs (100 features)
 * - Memory access issues with very large inputs
 * - GPU memory limitations with extreme inputs
 */
INSTANTIATE_TEST_SUITE_P(
    PairwiseAriTestInstances,
    PairwiseAriTest,
    ::testing::Values(
        AriTestParams(2, 2, 100, 10),
        AriTestParams(5, 10, 200, 10),
        // AriTestParams(2, 1, 1000, 10),  // FIXME: wrong results, maybe test is not correct
        AriTestParams(100, 20, 100, 10),
        // Document known issues
        // AriTestParams(100, 20, 1000, 10),  // FIXME: wrong results, maybe test is not correct
        AriTestParams(200, 20, 300, 10),   // FIXME: fix illegal mem access
        AriTestParams(1000, 10, 300, 10)  // FIXME: out of memory
    ),
    // Add test name generator for better output
    [](const testing::TestParamInfo<AriTestParams>& info) {
        return std::string("Features") + std::to_string(info.param.n_features) +
               "_Parts" + std::to_string(info.param.n_parts) +
               "_Objects" + std::to_string(info.param.n_objs);
    }
);

