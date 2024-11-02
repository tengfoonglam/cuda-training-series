cmake_minimum_required(VERSION 3.8)
project(cuda_training_series_hw7 LANGUAGES C CXX CUDA)

# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

set(use_streams 1)

foreach(example_name overlap multi)
    add_executable("${example_name}" "${example_name}.cu")
    add_executable("${example_name}_solution" "${example_name}_solution.cu")
    if("${example_name}" STREQUAL "overlap")
        message("Compiling ${PROJECT_NAME} example's ${example_name} with USE_STREAMS=${use_streams}")
        target_compile_definitions("${example_name}" PRIVATE USE_STREAMS=${use_streams})
        target_compile_definitions("${example_name}_solution" PRIVATE USE_STREAMS=${use_streams})
    endif()
endforeach()
