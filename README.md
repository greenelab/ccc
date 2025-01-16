# Clustermatch Correlation Coefficient GPU (CCC-GPU)

## Development
[Scikit-build](https://scikit-build-core.readthedocs.io/en/latest/getting_started.html) is used to build the C++ CUDA extension module and its tests.

### How to set up the development environment
At the root of the repository, run:
```
conda env create -f environment-cuda.yml
```

### How to activate the development environment
At the root of the repository, run:
```
source ./setup_dev.sh
```
It will activate the conda environment and set up PYTHONPATH for the current shell session.

This script can also be configured as a startup script in PyCharm so you don't have to run it manually every time.

### How to install this CUDA module
At the root of the repository, run:
```
conda activate ccc-cuda
# This will build the c++ module and install it with the Python package in the current environment
pip install .
```

### How to only build the C++ CUDA extension module and its tests
```
# Clean up the build directory
rm -rf build
# Read ./CMakeLists.txt, configure the project, generate the build system files in the ./build directory
cmake -S . -B build
# Compile the project, generate the executable files in the ./build directory
cmake --build build
```

### How to run C++ tests in tests/cuda_ext
The CMakeLists.txt file in the root directory will pick up the tests in tests/cuda_ext and build them.
```
for test in build/test_*; do
    echo "Running $test..."
    ./$test
done
```
