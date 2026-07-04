cmake_minimum_required(VERSION 3.16)
project(editor)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(tinyxml2 CONFIG REQUIRED)
find_package(DART REQUIRED COMPONENTS collision-bullet utils CONFIG)
find_package(boost_asio CONFIG REQUIRED)

file(GLOB editor_srcs "src/*.cpp" "src/*.h")

add_executable(editor ${editor_srcs})
target_compile_features(editor PRIVATE cxx_std_17)

target_link_libraries(editor PRIVATE
  sim                       # this project's simulation core (DART Environment, Character, NN)
  glfw
  glad::glad
  imgui::imgui
  imguizmo::imguizmo
  nlohmann_json::nlohmann_json
  tinyxml2::tinyxml2
  dart
  dart-collision-bullet
  dart-utils
  Boost::asio)

if(WIN32)
  target_link_libraries(editor PRIVATE opengl32 comdlg32 shell32 ws2_32 mswsock)
endif()

set_target_properties(editor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/editor)
