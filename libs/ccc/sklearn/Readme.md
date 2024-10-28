## How to build the CUDA module and its tests

```
# cd to current directory
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```
