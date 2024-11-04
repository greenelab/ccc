#pragma once

#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <typename T>
auto ari(const py::array_t<T, py::array::c_style>& parts, 
         const size_t n_features,
         const size_t n_parts,
         const size_t n_objs) -> std::vector<float>;
