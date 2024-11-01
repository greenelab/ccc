#pragma once

#include <vector>
#include <pybind11/numpy.h>

using Mat3 = std::vector<std::vector<std::vector<int>>>;

namespace py = pybind11;
// template <typename T>
std::vector<float> cudaAri(const py::array_t<int, py::array::c_style>& parts, const size_t n_features, const size_t n_parts, const size_t n_objs);
