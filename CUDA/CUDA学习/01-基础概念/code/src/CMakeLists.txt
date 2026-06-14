cmake_minimum_required(VERSION 3.28)
project(hello_world VERSION 1.0.0  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(hello_world week01_helloworld.cu)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(hello_world CUDA::cudart)