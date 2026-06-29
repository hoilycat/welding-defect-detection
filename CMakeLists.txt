cmake_minimum_required(VERSION 3.15)
project(weld_inspection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_compile_options(/utf-8)

add_executable(main src/main.cpp)
find_package(nlohmann_json REQUIRED)
target_link_libraries(main ${OpenCV_LIBS} nlohmann_json::nlohmann_json)