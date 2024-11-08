#include <iostream>
#include <vector>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    try {
        // Define vectors in C++
        std::vector<int> part0 = {2, 3, 6, 1, 0, 5, 4, 3, 6, 2};
        std::vector<int> part1 = {0, 6, 2, 5, 1, 3, 4, 6, 0, 2};

        // Import required Python modules
        py::module_ np = py::module_::import("numpy");
        py::module_ ccc_module = py::module_::import("ccc.sklearn.metrics");

        // Convert C++ vectors to numpy arrays
        py::array_t<int> np_part0 = py::cast(part0);
        py::array_t<int> np_part1 = py::cast(part1);

        // Call the ccc function
        py::object result = ccc_module.attr("adjusted_rand_index")(np_part0, np_part1);
        
        // Convert result to C++ double
        const auto correlation = result.cast<double>();

        std::cout << "ARI: " << correlation << std::endl;
    }
    catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}