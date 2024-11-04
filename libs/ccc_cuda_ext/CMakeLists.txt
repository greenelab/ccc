# CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(CudaAriProject LANGUAGES CUDA CXX)

# Set Python Standard
# Get Python version dynamically
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    OUTPUT_VARIABLE PYTHON_VERSION_FULL
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Set the paths using the detected version
set(Python_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)
set(PYTHON_INCLUDE_DIR $ENV{CONDA_PREFIX}/include/python${PYTHON_VERSION_FULL})
set(PYTHON_LIBRARY $ENV{CONDA_PREFIX}/lib/libpython${PYTHON_VERSION_FULL}.so)

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
set(PYBIND11_NEWPYTHON ON)
find_package(Python REQUIRED Development)
find_package(pybind11 CONFIG REQUIRED)

# # Add the CUDA library
# add_library(cudaAriLib STATIC metrics.cu)  # Add the CUDA source file
# set_target_properties(cudaAriLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Testing
enable_testing()

add_executable(testCudaAri tests/test_kernel.cpp metrics.cu)
target_link_libraries(testCudaAri PUBLIC GTest::gtest_main GTest::gtest pybind11::embed Python::Python)
gtest_discover_tests(testCudaAri)

# pybind11_add_module(testCudaAri tests/test_kernel.cpp)
