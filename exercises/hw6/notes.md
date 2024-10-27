# Lesson 6 Notes

## CUDA Unified Memory

- Standard CUDA 3-step processing sequence
  1. copy data from CPU to GPU
  2. Execute GPU computation
  3. copy output from GPU to CPU
- Would be great if we did not need to write code explicitly for 1. and 3.

#### CUDA 6+ Unified Memory

- Single allocation, single pointer to data accessible to both CPU and GPU
- Automatically migrates data to accessing processor while guarnteeing global coherence
- Unified memory is limited to the GPU memory size

#### CUDA 8+ Unified Memory

- Unified memory atomic operations
- Memory usage hints via cudaMemAdvise API
- Unified memory is limited to system memory size

#### Basic API

- `cudaMallocManaged` -> Allocate data in unified memory
- `cudaDeviceSynchronize` -> Copy GPU computed data to unified memory to be accessed by CPU
- `cudaFree` -> Free unified memory
- No need to use `cudaMalloc` and `cudaMemcpy` to manage on-device GPU data

#### Overview of how Unified Memory Works

- Page fault: Whenever new data needs to be written to a segment of reserved memory
- Memory is reserved for both CPU and GPU memeory
- Data is allocated in CPU (page faults triggered by `memset`)
- When GPU needs access to data, page fault occurs on GPU to cause transfer of the required data from CPU to GPU
- If data is accessed on CPU afterwards, page fault occurs (if data was processed by GPU) and is transferred back to CPU

#### Use Cases for Unified Memory

- Deep copy
  - Copy a struct which itself has an array (which is a pointer) to GPU
  - Each of these nested arrays also needs to have a new address on-device, which makes CUDA code bloated and complicated
- Linked list
  - Similar to deep copy where each element in the list needs to have its own on-device address
- C++ objects
  - Override `new` and `delete` to allocate data directly on unified memory, then pass data pointer direcly to kernel
- On-demand paging
  - Only required data is access by GPU, e.g. graph traversal only visited notes are transferred to GPU

#### Performance Considerations

- Every page fault triggers service overhead
- Extremely inefficient if moving large amounts of data, page-by-page
- For bulk data, "memcpy-like" operation is much more efficient

#### Tuning Methods
- Explicit prefetching of data using `cudaMemPrefetchAsync`
  - Can transfer data across any GPU and CPU (e.g. GPU to GPU is possible)
- Advise runtime on expected memory access behaviors using `cudaMemAdvise`
  - `cudaMemAdviseSetReadMostly`: Specify read duplication
    - Data will mainly be read-only
    - UM system will make a "local" copy for each processor, they are invalidated if a processor writes to it
  - `cudaMemAdviseSetPreferredLocation`: suggest best location
    - Suggests which processor is the best location for data
    - Data will be migrated to the preferred processor on demand
  - `cudaMemAdviseSetAccessedBy`: suggest mapping
    - Provide access to data without incurring page faults
    - Does not cause movement or affect location of data
    - Indicated processor receives a (P2P) mapping to the data
    - If the data is migrated, mapping is updated

#### UM Final Pointers

- UM is first and foremost about ease of programming and programmer productivity
- UM is not primarily a technique to make well-written CUDA codes run faster
- UM cannot do better than expertly written manual data movement, in most cases
- It can be harder to achieve expected concurrency behavior with UM
- Misuse of UM can slow a code down dramatically
- There are scenarios where UM may enable a design pattern (e.g. graph traversal)
- Oversubscription does not easily/magically give you GPU-type performance on arbitrary datasets/algorithms
- For codes that tend to use many different libraries, each of which makes some demand on GPU memory with no regard for what other libraries are doing, UM can sometimes be a primary way to tackle this challenge (via use of oversubscription), rather than an entire rewrite of the codebase

## HW 6 Notes

- Besides using Nvidia Visual Profiler, the following command can be run to execute performance analysis and print out the results in terminal

`nsys profile --stats=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true --cuda-memory-usage=true --show-output=true [terminal command to run application]`

#### Array Increment

| Experiment     | Kernet Execution Time / us | CPU Page Faults | GPU Page Faults | HtoD Operations | DtoH Operations |
| -------------- | -------------------------- | --------------- |---------------- | --------------- | --------------- |
| No UM          |         1902.86            |        -        |        -        |        1        |        1        |
| UM Naive       |         50631.31           |       768       |        0        |        5316     |        768      |
| UM Prefetching |         1901.53            |       384       |        0        |        64       |        64       |

- With prefetching, we have 'recovered' the performance lost using unified memory by prefetching the data before and after the kernel execution.
- Hence, it UM might make it easier to write CUDA code but it might take a performance hit if you are not careful with your implementation
- Note: Not sure why there are no GPU page faults for both UM implementations

#### Increasing execution to N=10000 instead of N=1

- As kernel size is small, for UM Prefetching case, `cudaMallocManaged` takes up 78.9% of the execution time
- When kernel is run 10000 times instead of once, `cudaMallocManaged` takes up 0.5% of the execution time which shows that the additional overhead of using UM might become negligible for expensive CUDA programs
