# Clustermatch Correlation Coefficient GPU (CCC-GPU)

## Development
### How to build the CUDA module and its tests

```
cmake -S . -B build
cmake --build build
```

### How to build and install this CUDA module
```
conda activate ccc-rapid
# This will build the c++ module and install it with the Python package in the current environment
pip install .
```

### How to run C++ tests in tests/cuda_ext
The CMakeLists.txt file in the root directory will pick up the tests in tests/cuda_ext and build them.
```
for test in build/test_ari{,_py,_random}; do
    echo "Running $test..."
    ./$test
done
```
