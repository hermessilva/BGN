cmake_minimum_required(VERSION 3.16)
project(gaitnet)

set(OpenGL_GL_PREFERENCE LEGACY)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT MSVC)
  link_libraries(stdc++fs)
  add_compile_options(-fPIC)
  add_compile_options(-Wdeprecated-declarations)
  add_compile_options(-DHAVE_CSTDDEF)
else()
  # MSVC: disable min/max macros, enable M_PI and lax preprocessor for DART headers
  add_compile_definitions(NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
  add_compile_options(/bigobj /permissive-)
endif()

add_definitions(-DMASS_ROOT_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")

# Build the OpenGL/ImGui viewer (needs a display). Off for headless/server builds.
option(GAITNET_BUILD_VIEWER "Build the GLFW/ImGui viewer" ON)
if(SERVER_BUILD)
  set(GAITNET_BUILD_VIEWER OFF)
endif()

add_subdirectory( sim )
add_subdirectory( python )
if(GAITNET_BUILD_VIEWER)
  add_subdirectory( libs )
  add_subdirectory( viewer )
endif()
