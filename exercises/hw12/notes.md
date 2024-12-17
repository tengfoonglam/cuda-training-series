# Lesson 12 - CUDA Debugging

## Basic CUDA Error Checking
- All [CUDA runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html) calls return an error code in the form of the enum type `cudaError_t`
- `cudaGetErrorString(cudaError_t err)` converts an error code to human-readable string
- Usual kernel launch syntax is not a CUDA runtime API call and does not return an error code

#### Asynchronous Errors
- CUDA kernel launches are asynchronous
  - Kernel may not begin executing right away, host thread continues without waiting for kernel to complete
- If a CUDA error is detected during kernel execution, it will be signalled in the next CUDA runtime API call, after the error is detected

#### Kernel Error Checking
- CUDA kernel launches can produce two types of errors
- **Synchronous**: Detectable right at launch
  - Detect right away using
    - `cudaGetLastError()` - Gets last error and clears it if it is non-sticky
    - `cudaPeekAtLastError()` - Gets last error but does not clear it
- **Asynchronous**: Occurs during device code execution
  - Requires tradeoffs to detect them
  - Use synchronizing calls like `ret = cudaDeviceSynchonize()` but this breaks concurrency
  - Debug macro
  - Set environment variable `CUDA_LAUNCH_BLOCKING` to 1 to globally disable asynchronicity of kernel launches

#### Sticky vs Non-Sticky Errors
- Non-sticky error is recoverable
  - Do not "corrupt the CUDA context"
  - Subsequent CUDA runtime API calls behave normally
  - Example: Out of memory error
- Sticky error is not recoverable
  - A result of kernel code execution error (e.g. Kernel time-out, illegal instruction, invalid address, etc)
  - CUDA runtime API no longer usable in that process
  - All subsequent CUDA runtime API calls will return the same error
  - Only "recovery" method is to terminate the host process
  - [Multi-process](https://stackoverflow.com/questions/56329377/reset-cuda-context-after-exception) application can be designed to allow recovery

#### Example Macro
- Use macro instead of function so it prints the line number where the error occured

```c++
#include <stdio.h>
#define cudaCheckErrors(msg) \
do { \
cudaError_t __err = cudaGetLastError(); \
if (__err != cudaSuccess) { \
fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
msg, cudaGetErrorString(__err), \
__FILE__, __LINE__); \
fprintf(stderr, "*** FAILED - ABORTING\n"); \
exit(1); \
} \
} while (0)
```

## Compute-Sanitizer Tool

#### Compute-Sanitizer
- Provides “automatic” runtime API error checking – even if your code doesn’t handle errors
- Works with various language bindings: CUDA Fortran, CUDA C++, CUDA Python, etc
- Subtools
  - **memcheck** (default): detects illegal code activity: illegal instructions, illegal memory access, misaligned access, etc
  - **racecheck**: detects shared memory race conditions/hazards: RAW, WAW, WAR
  - **initcheck**: detects accesses to global memory which has not been initialized
  - **synccheck**: detects illegal use of synchronization primitives (e.g. __syncthreads())

#### Memcheck Sub-Tool
- Basic usage: `compute-sanitizer ./my_executable`
- Recommended to run this tool first before other tools
- Detects kernel execution errors:
  - Invalid/out-of-bounds memory access
  - Invalid PC/Invalid instruction
  - Misaligned address for data load/store
- Also do leak checking for device-side memory allocation/free
- Provides error localization when your code is compiled with `–lineinfo`
- Has a performance impact on speed of kernel execution
- Error checking is “tighter” than ordinary runtime error checking

#### Racecheck Sub-Tool
- Basic usage: `compute-sanitizer --tool racecheck ./my_executable`
- Finds shared memory (only) race conditions:
  - WAW – two writes to the same location that don’t have intervening synchronization
  - RAW – a write, followed by a read to a particular location, without intervening synchronization
  - WAR – a read, followed by a write, without intervening synchronization

#### Syncheck Sub-Tool
-  Basic usage: compute-sanitizer --tool synccheck ./my_executable
- Applies to usage of `__syncthreads()`, `__syncwarp()`, and CG equivalents (e.g. `this_group.sync()`)
- Detects illegal use of syncrhonization, where not all necessary threads can reach the sync point at threadblock/warp level
- Ensures that mask parameter passed `__syncwarp()` is invalid
- Applicability is limited on cc 7.0 and beyond due to volta execution model relaxed requirements

## Debugging with CUDA-GDB
- Based on widely-used gdb debugging tool
- “command-line” debugger, allows for typical operations like:
  - setting breakpoints (e.g. `b`)
  - single-stepping (e.g. `s`)
  - inspection of data (e.g. `p`)

#### Building Debug Code
- Debug compile flags for NVCC
  - `-g` – standard gnu switch for building a debug (host) code
  - `-G` – builds debug device code
- This makes the necessary symbol information available to the debugger so that you can do “sourcelevel” debugging
- The `-G` switch has a substantial impact on device code generation. Use it for debug purposes only
  - Don’t do performance analysis on device code built with the `-G` switch
  - In rare cases, it may change the behaviour of your code
- Make sure your code is compiled for the correct target: e.g. `-arch=sm_70`

#### Additional Prep Suggestions
- Make sure your code completes the various sanitizer tool tests
- Make sure your host code is “sane” e.g. does not seg fault
- Make sure your kernels are actually being launched
  - `nsys profile --stats=true ./my_executable` and check CUDA Kernel statistics

#### CUDA Specific Commands
- set cuda … - used to set general options and advanced settings
  - launch_blocking (on/off) - make launches pause the host thread
  - break_on_launch (option) - break on every new kernel launch
- info cuda … - get general information on system configuration
  - devices, sms, warps, lanes, kernels, blocks, threads, …
- cuda … - used to inspect or set current focus
  - (cuda-gdb) cuda device sm warp lane block thread -display current focus coordinates
  - block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0
  - (cuda-gdb) cuda thread (15) - change coordinate(s)
