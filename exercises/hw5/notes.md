# Lesson 5 Notes

## Transformation Vs Reduction

- Transformation: Input and output have about the same size
  - One thread per output point
- Reduction: Output size significantly smaller than input size

## Atomics

- Read, modify and write in the same indivisible operation
- May have performance implications
- For available Cuda atomic operations, refer [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
- Uses
    1. Used to determine next work item, queue slot, etc
    2. Reserve space in buffer
      - If each thread produces a variable amount of data, create a buffer array and a buffer_idx to keep track of where incoming data should be written next (using atomic operations)

## Classical Parallel Reduction

- Using only atomics to perform reduction might make the atomic operations the bottleneck of the kernel implementation
- Should use all threads as much as possible
- Parallel reduction: Use threads to reduce the input over multiple iterations
- Example: Sum up pairs and get the final sum using a tree-based approach

#### Sequential Addressing

- Each thread is in charge of processing elements of its own thread idx and (thread idx + offset) --> two elements reduced to one
- offset is halved every iteration
- Sequential addressing is bank conflict free

#### Grid-Stride Loops

- Kernel that load and operate on arbitrary data sizes efficiently
  - Kernel is decoupled with grid size
  - Kernel should work whether # threads >/</= number of data elements
- Kernel must have number of threads equal to the specified grid width
- To ensure optimized coalesced load/stores of shared memory, each thread should operate over grid-width intervals at every iteration
- e.g. For 1D grid, `idx += gridDim.x * blockDim.x; // grid width`
  - Note: blockDim.x = number of # threads in a block
- Useful technique when first populating shared memory with data from input array

## Warp Shuffle

- Allows for intra-warp communication
- Both source and destination threads in the warp must “participate”
- Sync “mask” used to identify and reconverge needed threads
- Useful variables in code
  - Thread number within a warp `int lane = threadIdx.x % warpSize;`
  - Number of warps within a block `int warpID = threadIdx.x / warpSize;`
- Benefits
  - Reduce or eliminate shared memory usage
  - Single instruction instead of two or more instructions using shared memory
  - Reduce level of explicit synchronization

## Case Study: Sum All Elements in Array of size N

#### Without Warp Shuffle (# Threads: block size)
1. Create shared memory array of block size elements
2. Using grid-stride loop, compute intermediate sums (N/block_size elements) to shared memory
3. Using sequential addressing, add intermetide results iteratively
4. First thread of each block perform an atomic sum to the final output

#### With Warp Shuffle (# Threads: # of warps per thread block (maximum 1024 / 32 = 32))
1. Create shared memory array of equal to number of threads
2. Use grid stride technique to compute intermediate sum in local variable `val` (as opposed to storing in shared memory)
3. Using sequential addressing and warp shuffle, accumulate sum within each warp and write to shared memory
    - Hence each warp contributes one intermediate sum
4. Using the first warp of each block, sum all elements in shared memory using sequential addressing and warp shuffle
5. First thread of each block perform an atomic sum to the final output

## HW 5 Notes

#### Comparing Reductions

|    N    | Atomic/us | Parallel Reduction/us | Warp Shuffle Reduction/us |
| ------- | --------- | --------------------- | ------------------------- |
| 163840  | 365       | 23                    | 16                        |
| ~8M     | 18591     | 246                   | 236                       |
| ~16M    | 37320     | 453                   | 455                       |
| ~32M    | Incorrect sum | 906               | 905                       |

- Atomic reduction is the slowest for all N
- For >8M, there is not a big difference between parallel an warp shuffle reductions
- The minimal difference is because the memory bandwidth of the GPU became the limiting factor instead of the kernel implementation
   - According to this [website](https://www.techspot.com/review/1209-nvidia-geforce-gtx-1060/), the GTX 1060 has a theoretical memory bandwidth of 192gbps
   - Based on the N=~8M results, actual memory bandwidth is roughly
     (4 bytes/int * 8000000 ints) / (250 * 10 ^ -6 s) * (10^-9 GB / B) = 128 gbps
- At a smaller N, the warp shuffle is noticeably faster than parallel reduction (30% faster)
- For 32M case, there are not enough threads for each element for the input array so the sum was incorrect. Max number of threads 1024 * 1024 = 1638400
