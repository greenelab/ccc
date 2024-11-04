# CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(CudaAriProject LANGUAGES CUDA CXX)

# Add gtest as a dependency
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5ed21863955149a5a877a53d7d5045b6919090ed.zip
)
include(GoogleTest)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the CUDA library
add_library(cudaAriLib STATIC metrics.cu)  # Add the CUDA source file
set_target_properties(cudaAriLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Testing
enable_testing()

add_executable(testCudaAri tests/test_kernel.cpp)
target_link_libraries(testCudaAri PRIVATE cudaAriLib GTest::gtest_main GTest::gtest)
gtest_discover_tests(testCudaAri)
