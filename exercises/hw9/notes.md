# Lesson 9 Notes


## Coorperative Groups

- A flexible model for synchronization and communication within groups of threads
- `__syncthreads()` is only on the block level

## API Overview

- All Cooperative Groups functionality is within a `cooperative_groups:: namespace`

For current coalesced set of threads: `auto g = coalesced_threads();`
For warp-sized group of threads:
```C++
auto block = this_thread_block();
auto g = tiled_partition<32>(block); // Compile time known size
auto tile32 = tiled_partition(g, 32); // Runtime known size
thread_group tile4 = tiled_partition(tile32, 4); // Further decomposition of tile
```
- **Note**: Number of threads restricted to powers of two, and <= 32 in initial release
For CUDA thread blocks `auto g = this_thread_block();`
For device-spanning grid: `auto g = this_grid();`
For multiple grids spanning GPUs: `auto g = this_multi_grid();`

#### Thread Groups

- Implicit group of all the threads in the launched thread block
  - thread_group interface:
  - `void sync();` // Synchronize the threads in the group
  - `unsigned size();` // Total number of threads in the group
  - `unsigned thread_rank();` // Rank of the calling thread (i.e. like and identifying index) within between 0 and (size-1)
  - `bool is_valid();` // Whether the group violated any API constraints
  - Additional thread_block specific functions:
    - `dim3 group_index();` // 3-dimensional block index within the grid
    - `dim3 thread_index();` // 3-dimensional thread index within the block

#### Thread Block Tile

- A subset of threads of a thread block, divided into tiles in row-major order
- `thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());`
- Specifying tile size at compile time increases performance
- Exposes additional functionality as compared to its runtime counterparts
  - `.shfl()`
  - `.shfl_down()`
  - `.shfl_up()`
  - `.shfl_xor()`
  - `.any()`
  - `.all()`
  - `.ballot()`
  - `.match_any()`
  - `.match_all()`

#### Generic Parallel Algorithms

- Kernels can now be agnostic to the number of threads available for compute

Example 1
```C++
__device__ int reduce(thread_group g, int *x, int val) {
int lane = g.thread_rank();
for (int i = g.size()/2; i > 0; i /= 2) {
    x[lane] = val;
    g.sync();
    if (lane < i) val += x[lane + i];
    g.sync();
}
return val;
}
```

Example 2, if # threads known at compile time
```C++
template <unsigned size>
__device__ int tile_reduce(thread_block_tile<size> g, int val) {
for (int i = g.size()/2; i > 0; i /= 2) {
val += g.shfl_down(val, i);
}
return val;
}
```

#### Grid Group

- A set of threads within the same grid, guaranteed to be resident on the device
- `cudaLaunchCooperativeKernel(…)`
- Note: Device needs to support the `cooperativeLaunch` property.
  - `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, numThreads, 0);`
- This enables syncing of threads device wide, not just block wide
- Benefits:
  - In some cases, by being able to sync device wide, some algorithms can have multiple steps (kernels) that rely on computations from previous kernels no longer need to copy the intermediate data between host/device. Just sync the threads and proceed on to the next step in the same kernel while keeping the intermediate data on device

Example
```C++
__global__ kernel() {
grid_group grid = this_grid();
// load data
// loop - compute, share data
grid.sync();
// device wide execution barrier
}
```

#### Multi Grid Group
- A set of threads guaranteed to be resident on the same system, on multiple devices
- Opt-in: `cudaLaunchCooperativeKernelMultiDevice(…)`
- Note: Devices need to support the `cooperativeMultiDeviceLaunch` property.


```C++
__global__ void kernel() {
multi_grid_group multi_grid = this_multi_grid();
// load data
// loop - compute, share data
multi_grid.sync();
// devices are now synced, keep on computing
}
```

#### Coalesced Group
- Discover the set of coalesced threads, i.e. a group of converged threads executing in SIMD

```C++
coalesced_group active = coalesced_threads();
if(...)
{
    // This group will only contain threads that are in this if condition
    coalesced_group g1 = coalesced_threads();
    g1.thread_rank(); // Number of threads in this if condition `indexed`
    ...
}
```

```C++
// Each thread increments the argument int by 1
// Return argument should be the previous value of p before the increment of 1
// Example: If p has value 3, and there are 3 threads running the kernel
// then the threads should return 3, 4, 5
inline __device__ int atomicAggInc(int *p)
{
    coalesced_group g = coalesced_threads();
    int prev;
    if (g.thread_rank() == 0) {
        prev = atomicAdd(p, g.size());
        // Single atomic add to the value, instead of multiple atomic adds of value one
        // Note that atomicAdd does not return the updated value, instead it returns the old value
    }

    //shlf: Returns value of prev from thread 0
    //shlf implicitly forces a sync of all threads in the kernel
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}
```
