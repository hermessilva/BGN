find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

# glad OpenGL loader (vendored; cross-platform, uses WGL on Windows / GLX/EGL elsewhere)
add_library(glad STATIC glad/src/glad.c glad/include/glad/glad.h glad/include/KHR/khrplatform.h)
target_include_directories(glad PUBLIC glad/include)
target_link_libraries(glad PUBLIC OpenGL::GL)

# Dear ImGui + ImPlot (vendored). Uses the glad loader in imgui_impl_opengl3.
add_library(imgui STATIC
        imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_widgets.cpp imgui/imgui_demo.cpp
        imgui/examples/imgui_impl_glfw.cpp imgui/examples/imgui_impl_opengl3.cpp
        implot/implot.cpp implot/implot_demo.cpp implot/implot_items.cpp)
target_include_directories(imgui PUBLIC imgui implot)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(imgui PUBLIC glfw glad)
