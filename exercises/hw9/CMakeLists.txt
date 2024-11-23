cmake_minimum_required(VERSION 3.8)
project(cuda_training_series_hw9 LANGUAGES C CXX CUDA)

# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

foreach(example_name task1)
    add_executable("${example_name}_${PROJECT_NAME}" "${example_name}.cu")
    add_executable("${example_name}_solution_${PROJECT_NAME}" "${example_name}_solution.cu")
endforeach()
