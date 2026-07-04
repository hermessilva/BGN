#include "GLFWApp.h"
#include <pybind11/embed.h>
#include <iostream>

namespace py = pybind11;

int main(int argc, char **argv)
{
    // Keep the interpreter alive across the catch blocks: if it were destroyed
    // while unwinding (guard inside the try), any pybind call in the handler
    // would touch a finalized interpreter and crash in PyGILState_Ensure.
    pybind11::scoped_interpreter guard{};
    try
    {
        // Make the project's python/ package importable regardless of the current
        // working directory or PYTHONPATH (ray_model, forward_gaitnet, ...), plus
        // the build output dir that holds the pysim extension module.
        {
            py::object sys_path = py::module::import("sys").attr("path");
            std::string rootpy = std::string(MASS_ROOT_DIR) + "/python";
            sys_path.attr("insert")(0, rootpy);
            sys_path.attr("insert")(0, rootpy + "/Release");
            sys_path.attr("insert")(0, rootpy + "/Debug");
        }
        Environment *env = new Environment();
        GLFWApp app(argc, argv);
        app.setEnv(env);
        app.startLoop();
    }
    catch (pybind11::error_already_set &e)
    {
        std::cerr << "[viewer] python error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[viewer] fatal (" << typeid(e).name() << "): " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "[viewer] fatal: unknown exception" << std::endl;
        return 1;
    }
    return -1;
}
