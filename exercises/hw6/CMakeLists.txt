cmake_minimum_required(VERSION 3.22)
project(cuda_training_series_hw6 LANGUAGES C CXX CUDA)

# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

foreach(example_name linked_list array_inc)
    add_executable("${example_name}" "${example_name}.cu")
    add_executable("${example_name}_solution" "${example_name}_solution.cu")
endforeach()

add_executable(array_inc_original array_inc_original.cu)
