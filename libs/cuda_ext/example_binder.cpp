#include <pybind11/pybind11.h>
#include "example.hpp"

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_PLUGIN(_core) {
    py::module m("wrap", "pybind11 example plugin");
    m.def("add", &add, "A function which adds two numbers",
          "i"_a=1, "j"_a=2);
    return m.ptr();
}
