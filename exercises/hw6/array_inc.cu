#include <cstdio>
#include <cstdlib>
// error checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

template <typename T> void alloc_bytes(T &ptr, size_t num_bytes) {

  cudaMallocManaged(&ptr, num_bytes);
}

__global__ void inc(int *array, size_t n) {
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < n) {
    ++array[idx];
    idx += blockDim.x * gridDim.x; // grid-stride loop
  }
}

static constexpr size_t DS = 32ULL * 1024ULL * 1024ULL;

// Options to toggle between the different configurations required for the HW
// questions
static constexpr bool PREFETCH_DATA = true;
static constexpr int NUMBER_KERNEL_EXECUTIONS = 1; // 10000

int main() {

  int *um_array;
  alloc_bytes(um_array, DS * sizeof(um_array[0]));
  cudaCheckErrors("cudaMalloc Error");
  memset(um_array, 0, DS * sizeof(um_array[0]));
  if constexpr (PREFETCH_DATA) {
    cudaMemPrefetchAsync(um_array, DS * sizeof(um_array[0]), 0);
  }
  for (size_t i = 0; i < NUMBER_KERNEL_EXECUTIONS; ++i) {
    inc<<<256, 256>>>(um_array, DS);
  }
  cudaCheckErrors("kernel launch error");
  if constexpr (PREFETCH_DATA) {
    cudaMemPrefetchAsync(um_array, DS * sizeof(um_array[0]), cudaCpuDeviceId);
  }
  cudaDeviceSynchronize();
  cudaCheckErrors("cudaDeviceSynchronize Error");
  for (int i = 0; i < DS; ++i) {
    if (um_array[i] != NUMBER_KERNEL_EXECUTIONS) {
      printf("mismatch at %d, was: %d, expected: %d\n", i, um_array[i], 1);
      return -1;
    }
  }
  printf("success!\n");
  return 0;
}
