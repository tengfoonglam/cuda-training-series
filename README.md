# CUDA Training Resource

The material in this repository accompany the CUDA Training Series presented at ORNL and NERSC.

This is my personal attempt on the course exercises with complementary notes. As the original readme instructions require the student to run the exercises on NERSC's Cori GPU nodes, I provide the necessary CMakeLists/instructions to compile/run all exercise on a local laptop with a GPU installed.

You can find the slides and presentation recordings [here](https://www.olcf.ornl.gov/cuda-training-series/).
Original exercise repository can be found [here](https://github.com/olcf/cuda-training-series).

Besides the original contents, each HW folder contains:
- the completed exercises
- **notes\.md** with my personal summary of the lecture material + tips/results for the exercise the completed exercises
- **original** folder with uncompleted exercises in case you would like to try it yourself

**DISCLAIMER**: This is my attempt at the course so there may be undiscovered mistakes in the code/explanations.

## Installing CUDA on Ubuntu

- Follow the instructions [here](https://www.cherryservers.com/blog/install-cuda-ubuntu) to upgrade Nvidia drivers and install CUDA
- Install CMake: `sudo apt install cmake`
- Install [MPICH](https://www.mpich.org/) (required for HW11): `sudo apt install mpich`

In the event that you need to uninstall CUDA to install another version, run
```shell
sudo apt-get --allow-change-held-packages --purge remove "*cublas*" "cuda*" "nsight*"
```

As a reference, these are the system specifications that worked for me:
- GPU: NVIDIA GeForce GTX 1060
- Operating system: Ubuntu 22.04.1 LTS
- CUDA: 11.8
- CMake: 3.22.1

## Things that I did that Deferred from the Original Course

- Most of the profiling is done using Nvidia Visual Profiler instead of Nvidia NSight Compute as my GPU does not support NSight Compute.
- HW 10 has an exercise on multi-GPU, skipped as I do not have access to such as setup.

## Getting Nvidia Visual Profiler to work

Try to run the command `nvvp`

If Nvidia Visual Profiler does not run properly, apply the following fix [here](https://askubuntu.com/questions/1472456/cannot-open-nvidia-visualizer-profilernvvp).

Install OpenJDK
`sudo apt install openjdk-8-jdk`

Then run
`nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java`

## Build Instructions

All exercises are configured to be compiled using [CMake](https://cmake.org/cmake/help/latest/guide/tutorial/index.html).

In the git root directory

1. `mkdir build && cd build`
2. `cmake ../ && make -j$(($(nproc) - 1))`

Build executables should be found in the exercises folder within the build folder.

## Pre-Commit Hooks

This repository uses [pre-commit hooks](https://pre-commit.com/) to ensure that code is checked for simple issues before it is committed.

#### Pre-Commit Hooks Setup

* Install pre-commit: `sudo apt install pre-commit`
* Update pre-commit version to prevent this error `Type tag 'textproto' is not recognized`: `pip install --force-reinstall -U pre-commit`

#### Running the hooks

* Run hooks on every commit: `pre-commit install`
* Running the hooks manually: `pre-commit run --all`
