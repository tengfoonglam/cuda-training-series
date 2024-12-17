# Lesson 1 - Introduction to CUDA C++

**Host**: CPU and its memory
**Device**: GPU and its memory

## CUDA key functionality
 - Use GPU to parallelize compute-intensive functions and CPU to perform rest of the sequential code
 - PCIe or NVLink Bus is used to transfer data from CPU to/from GPU

## Terms and Syntax

 - **\__global__**
   - Runs on device, can be called from host or device code
   - Device functions processed by Nvidia compiler
   - Host functions (e.g. main()) processed by standard host compiler
 - **<<<>>>**
   - Triple angle brackets mark a call to device code (a.k.a kernel launch)
   - Parameters inside the triple angle brackets are the CUDA kernel execution configuration
 - **Device pointers**
   - usually with prefix d_
   - Point to GPU memory, not dereferenced in host code
 - **Host pointers**
   - Point to CPU memory, not dereferences in device code
 - **Simple CUDA API for handling device memory**
   - cudaMalloc()
   - cudaFree()
   - cudaMemcpy()

## Blocks and Threads

**add<<<N, M>>>()**
Execute N times in separate blocks (in parallel), each block with M threads

**blockIdx.x / .y/ .z**: Index into the array from 0,...,N-1
**threadIdx.x / .y / .z**: Block can be split into parallel threads

With M threads per block, a unique index for each thread is given by

```int index = threadIdx.x + blockIdx.x * blockDim.x;```

In the event that the number of jobs is not a friendly multiple of blockDim.x

```
#define N (2048*2048)
#define THREADS_PER_BLOCK 512

__global__ void add(int *a, int *b, int *c, int n) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
if (index < n)
c[index] = a[index] + b[index];
}

add<<<(N + M-1) / M,M>>>(d_a, d_b, d_c, N);
```

## Threads vs Blocks
Threads have mechanisms to communicate and synchronize as compared to blocks
