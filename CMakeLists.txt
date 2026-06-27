cmake_minimum_required(VERSION 3.15)
project(weld_inspection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(weld main.cpp)
target_link_libraries(weld ${OpenCV_LIBS})
