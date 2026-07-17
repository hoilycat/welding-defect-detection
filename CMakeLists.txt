cmake_minimum_required(VERSION 3.15)
project(weld_inspection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(main PRIVATE /utf-8)
endif()
