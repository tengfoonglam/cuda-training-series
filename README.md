# CUDA Training Resource
The materials in this repository accompany the CUDA Training Series presented at ORNL and NERSC.

You can find the slides and presentation recordings [here](https://www.olcf.ornl.gov/cuda-training-series/)

## Installing CUDA on Ubuntu

- Follow the instructions [here](https://www.cherryservers.com/blog/install-cuda-ubuntu) to upgrade Nvidia drivers and install CUDA
- Install CMake: `sudo apt install cmake`

## Building

In the git root directory

1. `mkdir build && cd build`
2. `cmake ../ && make -j$(($(nproc) - 1))`

Build executables should be found in the exercises folder within the build folder.

## Pre-Commit Hooks

This repository uses [pre-commit hooks](https://pre-commit.com/) to ensure that code is checked for simple issues before it is committed.

### Pre-Commit Hooks Setup

* Install pre-commit: `sudo apt install pre-commit`
* Update pre-commit version to prevent this error `Type tag 'textproto' is not recognized`: `pip install --force-reinstall -U pre-commit`

### Running the hooks

* Run hooks on every commit: `pre-commit install`
* Running the hooks manually: `pre-commit run --all`
