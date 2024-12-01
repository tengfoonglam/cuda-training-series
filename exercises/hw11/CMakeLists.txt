cmake_minimum_required(VERSION 3.8)
project(cuda_training_series_hw11 LANGUAGES C CXX CUDA)

# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

find_package(MPI REQUIRED)

add_executable(test test.cu)
target_include_directories(test PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(test ${MPI_CXX_LIBRARIES})
