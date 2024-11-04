#include "metrics.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(ccc_cuda_ext, m) {
    m.doc() = "CUDA extension module for CCC";
    m.def("ari", &cudaAri, "CUDA version of Adjusted Rand Index (ARI) calculation",
        "parts"_a, "n_features"_a, "n_parts"_a, "n_objs"_a);
}
