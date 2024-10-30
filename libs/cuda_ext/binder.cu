#include "metrics.cuh"
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_PLUGIN(cuda_ccc) {
    py::module m("cuda_ccc", "pybind11 example plugin");
    m.def("ari", &cudaAri, "CUDA version of Adjusted Rand Index (ARI) calculation",
        "parts"_a, "n_features"_a, "n_parts"_a, "n_objs"_a);
    return m.ptr();
}
