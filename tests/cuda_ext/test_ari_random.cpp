#include <pybind11/embed.h> // everything needed for embedding
namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    // py::module_ coef = py::module_::import("ccc.coef");
    // py::module_ np = py::module_::import("numpy");

    py::exec(R"(
        from ccc.coef import ccc
        import numpy as np
        part0 = np.array([2, 3, 6, 1, 0, 5, 4, 3, 6, 2])
        part1 = np.array([0, 6, 2, 5, 1, 3, 4, 6, 0, 2])
        c = ccc(part0, part1)
        print(c)
    )");
}
