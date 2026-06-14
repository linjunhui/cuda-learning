cmake_minimum_required(VERSION 3.10)
project(OpenCVDemo)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(OpenCVDemo main.cpp)

target_link_libraries(OpenCVDemo ${OpenCV_LIBS})

