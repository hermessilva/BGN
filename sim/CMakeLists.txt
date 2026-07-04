cmake_minimum_required(VERSION 3.16)
project(sim)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

find_package(DART REQUIRED COMPONENTS collision-bullet utils CONFIG)
find_package(tinyxml2 CONFIG REQUIRED)

# Character.h embeds a Python interpreter (pybind11::embed) to evaluate the
# muscle / cascading networks, so the sim library needs Python + pybind11.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB srcs "*.h" "*.cpp")

add_library(sim STATIC ${srcs})
target_include_directories(sim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sim
  PUBLIC
    dart
    dart-collision-bullet
    dart-utils
    tinyxml2::tinyxml2
    pybind11::embed)
