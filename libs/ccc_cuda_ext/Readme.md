## How to build the CUDA module and its tests

```
# cd to libs/ccc_cuda_ext
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## How to build and install this CUDA module
```
conda activate ccc-rapid
pip install .

# This will build the c++ module and install it in the current environment
```

## How to run C++ tests in tests/cuda_ext
The CMakeLists.txt file in the root directory will pick up the tests in tests/cuda_ext and build them.

```
for test in build/test_ari{,_py,_random}; do
    echo "Running $test..."
    ./$test
done
```