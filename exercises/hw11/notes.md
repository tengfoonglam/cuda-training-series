# Lesson 11 Notes

#### Terminology
- MPI: Message Passing Interface -> Message passing standard in parallel computing architectures
- Rank: In MPI, each process is given its own rank (numeric IDs)

#### GPU Compute Modes
- Default: Multiple processes running at one time
- Exclusive process: Only one process can run at one time
- Prohibited: No process can run
- To change mode: `nvidia-smi --compute-mode`

#### Simple Oversubscription
- Case 1: Run entire task in a single process
- Case 2 (simple oversubscription): Break task down and split into multiple MPI ranks and run "concurrently"
  - While each rank operates fully independently, on the GPU, each individual process operates in time slices and a performance penalty is paid for switching between the time slices
  - i.e. GPU is only running one process at a single time and switching between the processes, hence there is no performance benefit
- No free lunch theorem: If GPU is fully utilised, cannot get faster answers
- If GPU is not fully utilised
  - Rarely simple oversubscription is beneficial
  - Typcially performs better when there is CPU-only work to interleave

#### Cuda Contexts
- Context is a stateful object required to run CUDA
- Automatically created when you use the CUDA runtime API
- on V100, each context is ~300MB + GPU code size, which limits how many contexts can fit on the GPU at a given time

#### Alternative Scheduling Methods
- Pre-emptive scheduling: Processes share GPU through time-slicing. Scheduling managed by system (as compared to oversubscription where GPU switches between the processes in a non-deterministic fashion)
- Concurrent scheduling: Processes run on GPU simultaneously. User creates and manages scheduling streams

#### Nvidia Multi-Process Service (MPS)
- Allows multiple processes to (instantaneously) share GPU resources (SMs)
- Designed to concurrently map multiple MPI ranks onto a single GPU
- Used when each rank is too small to fill the GPU on its own
- Oversubsription with MPS
  - MPS recovers performance losses due to context switching
  - But there is no significant speedup either
  - Performance increase from MPS does not outweigh incurred MPS overhead for small compute tasks
- In general
  - Strive to write your application so that you don’t need MPS
  - If you are unable to write kernels that fully saturate the GPU, then consider oversubscription, and MPS is usually always worth turning on for that case
  - Profile your code to understand why MPS did or did not help

#### Pre vs Post Volta MPS
- More MPS clients per GPU: 48 instead of 16
- **Less overhead**: Volta MPS clients submit work directly to the GPU without passing through the MPS server.
- **More security**: Each Volta MPS client owns its own GPU address space instead of sharing GPU address space with all other MPS clients.
- **More control**: Volta MPS supports limited execution resource provisioning for Quality of Service (QoS). -> CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
- **Independent work submission**: Each process has private work queues, allowing concurrent submission without contending over locks.

#### Using MPS

- No application modifications necessary
- MPU control daemon spawns MPS server upon CUDA application startup
- Debugging/profiling
  - Profiling tools are MPS-aware
  - `cuda-gdb` does nott support attaching but you can dump core files

```bash
nvidia-smi -c EXCLUSIVE_PROCESS # Ensure only one MPS server using GPU
nvidia-cuda-mps-control -d
```

#### MPS Environment Variables
- **CUDA_VISIBLE_DEVICES**: Sets devices which an application can see.
When set on MPS daemon, limits visible GPUs for all clients.
- **CUDA_MPS_PIPE_DIRECTORY** :Directory where MPS control daemon pipes are created. Clients & daemon must set to same value. Default is /var/log/nvidia-mps.
- **CUDA_MPS_LOG_DIRECTORY**: Directory where MPS control daemon log is
created. Default is /tmp/nvidia-mps.
- **CUDA_DEVICE_MAX_CONNECTIONS**: Sets number of hardware work queues that CUDA streams map to. MPS clients all share the same pool, so if set in an MPS-attached process sets this it may limit the max number
of MPS processes.
- **CUDA_MPS_ACTIVE_THREAD_PERCENTAGE**: Controls what fraction of GPU may be used by a process.

#### More on Execution Resource Provisioning with *CUDA_MPS_ACTIVE_THREAD_PERCENTAGE**
- Guarantees a process will use at most X percentage execution resources (SMs)
- Over-provisioning is permitted: sum across all MPS processes may exceed 100%
- Provisions only execution resources (SMs) – does not provision memory bandwidth or capacity
- Before CUDA 11.2, all processes be set to the same percentage
- Since CUDA 11.2, percentage may be different for each process
- Example with three processes:
  - With fractional provisioning: each process is guranteed 33% of the space. If a process needs more, it is still limited to 33%. If a process needs less, the remainder reserved space is unused
  - With oversubscription: Process that needs more space will fill up all the available space and other processes will have to wait for resources

#### Additional Considerations
1. **Memory footprint**
    - To provide a per-thread stack, CUDA reserves 1kB of GPU memory per thread
    - This is (2048 threads per SM x 1kB per thread) = 2 MB per SM used, or 164 MB per client for V100 (221 MB for A100)
    - **`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`** reduces max SM usage, and so reduces memory footprint
    - Each MPS process also uploads a new copy of the executable code, which adds to the memory footprint
1. **Work Queue Sharing**
    - CUDA maps streams onto **`CUDA_DEVICE_MAX_CONNECTIONS`** hardware work queues
    - Queues are normally per-process, but MPS allows 96 hardware queues to be shared among up to 48 clients
    - MPS automatically reduces connections-per-client unless environment variable is set
    - If **`CUDA_DEVICE_MAX_CONNECTIONS`** is set (e.g. to enable more concurrency within a process), this can reduce the maximum number of concurrent clients

#### Multi-Instance GPU (MIG)

- Divide a single A100 GPU into multiple instances, each with isolated paths through the entire memory system
- Benefits
  **1. Up to 7 GPU instances in a single A100**
       - Full software stack enabled on each instance, with dedicated SM, memory, L2 cache & bandwidth
  **2. Simultaneous workload execution with guaranteed quality Of service**
       - All MIG instances run in parallel with predictable throughput & latency, fault & error isolation
  **3. Diverse deployment environments**
       - Supported with bare metal, Docker, Kubernetes Pod, virtualized environments

#### Streams vs MPS vs MIG
- MPS
  - Dynamic contention for GPU resources
  - Single tenant
- MIG
  - Hierarchy of instances with guaranteed resrouce allocation
  - Multiple tenants

|                          | Streams        | MPS            | MPI         |
| ------------------------ | -------------- | -------------- |------------ |
| Partition Type           | Single process | Logical        | Physical    |
| Max Partitions           | Unlimited      | 48             | 7           |
| Performance Isolation    | No             | By percentage  | Yes         |
| Memory Protection        | No             | Yes            | Yes         |
| Memory Bandwidth QoS     | No             | No             | Yes         |
| Error Isolation          | No             | No             | Yes         |
| Cross-Partition Interop  | Always         | IPC            | Limited IPC |
| Reconfigure              | Dynamic        | Process launch | When idle   |

## HW 11 Notes

#### Setup

1. Need to install MPI to successfully compile executable
```shell
sudo apt install mpich
```

2. Before running executable, set the following environment variable
```shell
export RDMAV_FORK_SAFE=1
```

If not, the following error message will appear and the terminal will hang

```
A process has executed an operation involving a call
to the fork() system call to create a child process.

As a result, the libfabric EFA provider is operating in
a condition that could result in memory corruption or
other system errors...
```

3. In the build folder, run the following commands according to the instructions in the original [README](./README.md)

1 Rank execution
```shell
nsys profile --stats=true --show-output=true -t nvtx,cuda -s none -o 1_rank_no_MPS_N_1e9 -f true mpirun -np 1 ./exercises/hw11/test 1073741824
```

4 Rank execution
```shell
nsys profile --stats=true --show-output=true -t nvtx,cuda -s none -o 1_rank_no_MPS_N_1e9 -f true mpirun -np 4 ./exercises/hw11/test 1073741824
```

Turn on MPS
```shell
nvidia-cuda-mps-control -d
```

Turn off MPS
```shell
echo "quit" | nvidia-cuda-mps-control
```

#### Results


1 Rank execution (without MPS)
```
Time per kernel = 4.297 us
```

4 Rank execution (without MPS)
```
Time per kernel = 11.958 us
Time per kernel = 13.378 us
Time per kernel = 59816.6 us
Time per kernel = 59819.5 us
```

4 Rank execution (with MPS)
```
Time per kernel = 0.073 us
Time per kernel = 0.065 us
Time per kernel = 0.071 us
Time per kernel = 0.102 us
```

As expected, without MPS, running with simple oversubscription underperforms running with a single thread. This is most likely from the performance overhead of the GPU switching between the various time slices of the different ranks.

For my system, it seems that already at N=1073741824 running with MPS significantly outperforms both oversubscription and single process executable without MPS.
