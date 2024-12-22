cmake_minimum_required(VERSION 3.22)
project(cuda_training_series LANGUAGES C CXX CUDA)

foreach(hw_num RANGE 1 12)
    add_subdirectory(exercises/hw${hw_num})
endforeach()
