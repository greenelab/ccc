# CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(CudaAriProject LANGUAGES CUDA CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add the CUDA library
add_library(cudaAriLib STATIC metrics.cu)  # Add the CUDA source file
set_target_properties(cudaAriLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add the test executable
add_executable(testCudaAri test_kernel.cpp)
target_link_libraries(testCudaAri PRIVATE cudaAriLib)
