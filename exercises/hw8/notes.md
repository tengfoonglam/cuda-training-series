# Lesson 8 Notes

## Top-Level Performance Coding Objectives

- Make efficient use of the memory subsystem
  - Efficient use of global memory (coalesced access)
  - Intelligent use of the memory hierarchy
    - Shared, constant, texture, caches, etc
- Expose enough parallelism to saturate the machine and hide latency
  - Threads/blocks
  - Occupancy
  - Work per thread
  - Execution efficiency

## Analysis Driven Optimisation

 - Cycle
    1. Profile
    2. Determine limiter
    3. Inspect
    4. Optimize
    5. Repeat 1-4
 - Limiting behavior of a code may change over the duration of its execution cycle
 - Analyse small sections of code at at time (e.g. a single kernel)

## Types of Performance Limiters

- **Memory Bound**
  - Measured memory system performance is at or close to the expected maximum. (saturate memory bus)
- **Compute Bound**
  - The compute instruction throughput is at or close to the expected maximum.
- **Latency Bound**
  - One of the indicators for a latency bound code is when neither of the above are true

## Metrics for Determining Compute vs Memory Bound

 - Latency metrics
   - **sm efficiency**: smsp__cycles_active.avg.pct_of_peak_sustained_elapsed
 - Memory metrics:
   - **dram utilization**: dram__throughput.avg.pct_of_peak_sustained_elapsed
   - **L2 utilization**: lts__t_sectors.avg.pct_of_peak_sustained_elapsed
   - **shared utilization**: l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed
 - Compute metrics:
   - **DP Utilization**: smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active
   - **SP Utilization**: smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
   - **HP Utilization**: smsp__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active
   - **TC Utilization**: sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
 - **Integer Utilization**:
   - smsp__sass_thread_inst_executed_op_integer_pred_on.avg.pct_of_peak_sustained_active

## Memory Bound Limiter

- A code can be memory bound when either it is limited by memory bandwidth or latency (note: lump memory latency bound code with general latency case)
- Aim to optimize usage of the various memory subsystems, taking advantage of the memory hierarchy where possible
  - Optimize use of global memory
  - Under data reuse scenarios, make (efficient) use of higher levels of the memory hierarchy, and optimize these usages (L2 cache, shared memory)
  - Take advantage of cache “diversification” using special GPU caches – constant cache, read-only cache, texture cache/memory, surface memory
- For a code that is memory bandwidth bound, we can compute the actual throughput vs. peak theoretical

## Compute Bound Limiter

- A code is compute bound when the performance of a particular type of compute instruction/operation is at or near the limit of the functional unit servicing that type
- Aim to optimize the use of that functional unit type, as well as (possibly) seeking to shift the compute load to other types
- For a code that is dominated by a particular type (e.g. single precision floating point multiply/add) we can compare the actual throughput vs. peak theoretical

## Latency Bound Limiter

- A code is latency bound when the GPU cannot keep busy with the available/exposed parallel work
- Aim to expose more parallel work
  - Make sure to launch a large number of threads
  - Increase the work per thread (e.g. via a loop over input elements)
  - Use “vector load” to allow a single thread to process multiple input elements
  - Strive for maximum **occupancy**

## Occupancy

- A measure of the actual thread load in an SM, vs. peak theoretical/peak achievable
- CUDA includes an occupancy calculator spreadsheet
- Higher occupancy is sometimes a path to higher performance
- Achievable occupancy is affected by limiters to occupancy
- Primary limiters
  - Registers per thread (can be reported by the profiler, or can get at compile time)
  - Threads per threadblock
  - Shared memory usage

## HW 8 Notes

#### Profiling Setup

Some GPU devices are not supported by Nvidia NSight Compute. To check whether your GPU is compatible

1. Run `nv-nsight-cu-cli --list-chips`
2. Run `nvidia-smi` to check your GPU model
3. Search online the **code name** of your GPU model. For example, for the GeForce 10 series GPUs, the code names can be found [here](https://en.wikipedia.org/wiki/GeForce_10_series)
4. Ensure that the first ~5 characters of the code name is included in the chips listed in step 1. If not, the commands to do the profiling in the readme will not work

In the case NSight Compute does not work for your GPU (such as for me), you can use Nvidia Visual Profiler to run some profiling to get enough metrics to complete the exercise

1. After doing an initial profiling of the executable, in the **Analysis** tab, select **Unguided Analysis** and run **Kernel Performance**

#### Task 1 (Naive CUDA Transpose)

Duration: 525.530ms
Global Load Efficiency: 25%
Global Store Efficiency: 100%
Shared Efficiency: N/A

- Low load efficiency because there is no coalesced memory access
- Additionally, no shared memory is used

#### Task 2 (Shared Memory Transpose)

Duration: 2.342ms
Global Load Efficiency: 100%
Global Store Efficiency: 100%
Shared Efficiency: 11.8%

- Significant speedup, but shared memory efficiency is low, mainly because of large number of bank conflicts when accessing the shared memory

#### Task 3 (Padded Shared Memory Transpose)

Duration: 1.783ms
Global Load Efficiency: 100%
Global Store Efficiency: 99.1%
Shared Efficiency: 100%

- Adding an additional column reduced bank conflicts

###### Explanation of Bank Conflicts
Sources [here](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) and [here](https://stackoverflow.com/questions/15056842/when-is-padding-for-shared-memory-really-required)

To achieve high memory bandwidth for concurrent accesses, shared memory is divided into equally sized memory modules (banks) that can be accessed simultaneously. Therefore, any memory load or store of n addresses that spans b distinct memory banks can be serviced simultaneously, yielding an effective bandwidth that is b times as high as the bandwidth of a single bank.

However, if multiple threads’ requested addresses map to the same memory bank, the accesses are serialized. The hardware splits a conflicting memory request into as many separate conflict-free requests as necessary, decreasing the effective bandwidth by a factor equal to the number of colliding memory requests. An exception is the case where all threads in a warp address the same shared memory address, resulting in a broadcast. Devices of compute capability 2.0 and higher have the additional ability to multicast shared memory accesses, meaning that multiple accesses to the same location by any number of threads within a warp are served simultaneously.

To minimize bank conflicts, it is important to understand how memory addresses map to memory banks. Shared memory banks are organized such that successive 32-bit words are assigned to successive banks and the bandwidth is 32 bits per bank per clock cycle. For devices of compute capability 1.x, the warp size is 32 threads and the number of banks is 16. A shared memory request for a warp is split into one request for the first half of the warp and one request for the second half of the warp. Note that no bank conflict occurs if only one memory location per bank is accessed by a half warp of threads.

In general shared memory bank conflicts can occur any time two different threads are attempting to access (from the same kernel instruction) locations within shared memory for which the lower 4 (pre-cc2.0 devices) or 5 bits (cc2.0 and newer devices) of the address are the same. When a bank conflict does occur, the shared memory system serializes accesses to locations that are in the same bank, thus reducing performance. Padding attempts to avoid this for some access patterns. Note that for cc2.0 and newer, if all the bits are the same (i.e. same location) this does not cause a bank conflict.

An example
```
__shared__ int A[2048];
int my;
my = A[0]; // A[0] is in bank 0
my = A[1]; // A[1] is in bank 1
my = A[2]; // A[2] is in bank 2
...
my = A[31]; // A[31] is in bank 31 (cc2.0 or newer device)
my = A[32]; // A[32] is in bank 0
my = A[33]; // A[33] is in bank 1
```

Then if we access the data "row-wise", we will experience bank conflicts
```
my = A[threadIdx.x];    // no bank conflicts or serialization - handled in one trans.
my = A[threadIdx.x*2];  // 2-way bank conflicts - will cause 2 level serialization
my = A[threadIdx.x*32]; // 32-way bank conflicts - will cause 32 level serialization
```

Let's take a closer look at the 2-way bank conflict above. Since we are multiplying threadIdx.x by 2, thread 0 accesses location 0 in bank 0 but thread 16 accesses location 32 which is also in bank 0, thus creating a bank conflict. For the 32-way example above, all the addresses correspond to bank 0. Thus 32 transactions to shared memory must occur to satisfy this request, as they are all serialized.

Hence, if we know that the access pattern would be like this:

```
my = A[threadIdx.x*32];
```

Then we want to pad the data storage so that A[32] is a dummy/pad location to get the data with no bank conflicts.
