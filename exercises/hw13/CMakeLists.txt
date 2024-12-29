cmake_minimum_required(VERSION 3.22)
project(cuda_training_series_hw13 LANGUAGES C CXX CUDA)

# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()

add_executable("axpy_stream_capture_with_fixme" "axpy_stream_capture_with_fixme.cu")
add_executable("axpy_stream_capture_timer" "axpy_stream_capture_timer.cu")
add_executable("axpy_stream_capture_from_scratch" "axpy_stream_capture_from_scratch.cu")
add_executable("axpy_stream_capture_solution" "Solutions/axpy_stream_capture_solution.cu")

add_executable("axpy_cublas_with_fixme" "axpy_cublas_with_fixme.cu")
target_link_libraries("axpy_cublas_with_fixme" -lcublas)
add_executable("axpy_cublas_from_scratch" "axpy_cublas_from_scratch.cu")
target_link_libraries("axpy_cublas_from_scratch" -lcublas)
add_executable("axpy_cublas_solution" "Solutions/axpy_cublas_solution.cu")
target_link_libraries("axpy_cublas_solution" -lcublas)
