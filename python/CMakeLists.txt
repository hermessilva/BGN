cmake_minimum_required(VERSION 3.16)
project(pysim)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

find_package(OpenMP)
find_package(DART REQUIRED COMPONENTS collision-bullet utils CONFIG)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB srcs "*.h" "*.cpp")

pybind11_add_module(pysim ${srcs})
target_link_libraries(pysim
  PRIVATE
    sim
    dart
    dart-collision-bullet
    dart-utils)

if(OpenMP_CXX_FOUND)
  target_link_libraries(pysim PRIVATE OpenMP::OpenMP_CXX)
endif()
