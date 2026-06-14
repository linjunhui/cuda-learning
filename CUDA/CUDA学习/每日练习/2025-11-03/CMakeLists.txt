
cmake_minimum_required(VERSION 3.4)
project(device_count VERSION 1.0.0 LANGUAGES CUDA)

set(EXE_SOURCES device_count.cu)

add_executable(device_count ${EXE_SOURCES})

find_package(CUDAToolkit REQUIRED)

# target_link_libraries(device_count PRIVATE CUDA::cudart)
target_link_libraries(device_count ${CUDA_CUDART_LIBRARY})