#include "GLFWApp.h"
#include <iostream>

int main(int argc, char **argv)
{
    try
    {
        Environment *env = new Environment();
        GLFWApp app(argc, argv);
        app.setEnv(env);
        app.startLoop();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[viewer] fatal: " << e.what() << std::endl;
        return 1;
    }
    return -1;
}
