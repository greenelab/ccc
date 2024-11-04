cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CUDA CXX)

# Add this near the top of your file, after project()
# Define the include directories for the whole project
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/libs)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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


# Set CUDA architecture and Pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Download and configure Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2  # Adjust version as needed
)
FetchContent_MakeAvailable(googletest)

# Setup Gtest
enable_testing()
# Function to automatically add tests from a directory
function(add_tests_from_directory TEST_DIR)
    # Find all test files in the directory
    file(GLOB_RECURSE TEST_FILES 
        "${TEST_DIR}/*_test.cpp"    # Files ending with _test.cpp
        "${TEST_DIR}/*_tests.cpp"   # Files ending with _tests.cpp
        "${TEST_DIR}/test_*.cpp"    # Files starting with test_
    )

    # Loop through each test file
    foreach(TEST_FILE ${TEST_FILES})
        # Get the filename without extension
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        
        # Create an executable for this test
        add_executable(${TEST_NAME} ${TEST_FILE})

        # target_include_directories(${TEST_NAME} PRIVATE
        #     ${PROJECT_INCLUDE_DIR}  # Add this line
        #     ${Python_INCLUDE_DIRS}
        # )
        
        # Link against gtest and your project libraries
        target_link_libraries(${TEST_NAME} PRIVATE
            GTest::gtest_main
            GTest::gtest
            pybind11::headers
            Python::Module
            # Add your other project libraries here
            # project_lib
        )

        # Add the test to CTest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        
        # Set test properties (optional)
        # Set test properties (optional)
        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 10  # Timeout in seconds
            WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        )
    endforeach()
endfunction()

# Specify your test directory and call the function
# add_tests_from_directory(${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Optional: Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set up binding then do compilation and installation
set(CUDA_EXT_MODULE_NAME ccc_cuda_ext)
set(CUDA_EXT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/${CUDA_EXT_MODULE_NAME})
pybind11_add_module(${CUDA_EXT_MODULE_NAME} ${CUDA_EXT_DIR}/binder.cu ${CUDA_EXT_DIR}/metrics.cu)

# install(TARGETS ${CUDA_EXT_MODULE_NAME} LIBRARY DESTINATION .)
