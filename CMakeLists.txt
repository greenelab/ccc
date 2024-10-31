cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CUDA)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cuda_ccc libs/cuda_ext/binder.cu libs/cuda_ext/metrics.cu)

install(TARGETS cuda_ccc LIBRARY DESTINATION .)
