cmake_minimum_required(VERSION 3.16)
project(mysim)

if(NOT MSVC)
  add_compile_options(-fvisibility=hidden)
endif()

include_directories(../sim/)

find_package(DART REQUIRED COMPONENTS gui collision-bullet utils CONFIG)
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(FreeGLUT CONFIG REQUIRED)     # only for glutBitmapCharacter text
find_package(assimp CONFIG REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB srcs "*.h" "*.cpp")

add_executable(viewer ${srcs})
target_link_libraries(viewer
  PRIVATE
    sim
    imgui
    glad
    glfw
    dart
    dart-gui
    dart-collision-bullet
    dart-utils
    assimp::assimp
    OpenGL::GL
    OpenGL::GLU
    FreeGLUT::freeglut
    pybind11::embed)

set_target_properties(viewer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/viewer)
