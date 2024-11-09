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
