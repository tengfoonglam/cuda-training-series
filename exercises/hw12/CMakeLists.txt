cmake_minimum_required(VERSION 3.22)
project(cuda_training_series_hw12 LANGUAGES C CXX CUDA)

# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

add_executable("task1_${PROJECT_NAME}" "task1.cu")
target_compile_options("task1_${PROJECT_NAME}"
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
add_executable("task1_solution_${PROJECT_NAME}" "task1_solution.cu")
target_compile_options("task1_solution_${PROJECT_NAME}"
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

add_executable("task2_${PROJECT_NAME}" "task2.cu")
target_compile_options("task2_${PROJECT_NAME}"
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
add_executable("task2_solution_${PROJECT_NAME}" "task2_solution.cu")
target_compile_options("task2_solution_${PROJECT_NAME}"
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g;-G>)
