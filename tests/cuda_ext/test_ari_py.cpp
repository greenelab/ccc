#include <iostream>
#include <vector>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/stl.h>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::object scope = py::module_::import("__main__").attr("__dict__");
    py::exec(R"(
        from ccc.coef import ccc
        import numpy as np
        data = np.random.rand(5, 100)
        c = ccc(data)
    )");
    const auto result = py::eval("c", scope).cast<std::vector<float>>();;
    // Print the results
    std::cout << "Results: ";
    for (const auto& val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
