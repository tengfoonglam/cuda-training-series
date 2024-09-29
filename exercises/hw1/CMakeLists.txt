cmake_minimum_required(VERSION 3.8)
project(cuda_training_series_hw1 LANGUAGES C CXX CUDA)

# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

add_executable(hello_solution hello_solution.cu)
add_executable(hello hello.cu)
