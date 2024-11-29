cmake_minimum_required(VERSION 3.8)
project(cuda_training_series_hw10 LANGUAGES C CXX CUDA)

# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

find_package(OpenMP REQUIRED)

set(use_streams 1)

foreach(example_name streams)
    add_executable("${example_name}" "${example_name}.cu")
    target_link_libraries("${example_name}" OpenMP::OpenMP_CXX)
    target_compile_definitions("${example_name}" PRIVATE USE_STREAMS=${use_streams})
    add_executable("${example_name}_solution" "${example_name}_solution.cu")
    target_link_libraries("${example_name}_solution" OpenMP::OpenMP_CXX)
    target_compile_definitions("${example_name}_solution" PRIVATE USE_STREAMS=${use_streams})
endforeach()
