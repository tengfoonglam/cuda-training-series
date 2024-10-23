cmake_minimum_required(VERSION 3.8)
project(cuda_training_series LANGUAGES C CXX CUDA)

foreach(hw_num RANGE 1 5)
    add_subdirectory(exercises/hw${hw_num})
endforeach()
