#include <gtest/gtest.h>
#include <pybind11/numpy.h>
// #include "../metrics.cuh"

namespace py = pybind11;

TEST(AriTest, SimpleCase) {
    // Create input data
    std::vector<int> data = {
        0, 0, 1, 2,  // First partition
        0, 0, 1, 1   // Second partition
    };
    
    // Create shape and strides for 3D array (n_features=2, n_parts=1, n_objs=4)
    std::vector<ssize_t> shape = {2, 1, 4};
    // std::vector<ssize_t> strides = {4 * sizeof(int),  // stride for features
    //                                4 * sizeof(int),  // stride for partitions
    //                                sizeof(int)};     // stride for objects
    
    // // Create numpy array from data
    // py::array_t<int> parts(shape, strides, data.data());
    py::array_t<double> arr({ 3, 5 });
    
    // Call the ari function
    // std::vector<float> result = ari<int>(parts, 2, 1, 4);
    
    // Check result
    // ASSERT_EQ(result.size(), 1);  // Should only have one ARI value
    // EXPECT_NEAR(result[0], 0.57f, 1e-2);  // Compare with expected value within tolerance
}