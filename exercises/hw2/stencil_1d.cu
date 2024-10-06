#include <algorithm>
#include <stdio.h>

using namespace std;

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16
#define PADDED_ARRAY_SIZE 2 * RADIUS + N

__global__ void stencil_1d(int *in, int *out) {
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;

  // Index of current input value (with gindex) in temp store
  // Note temp is padded on both sides with RADIUS
  int lindex = threadIdx.x + RADIUS;

  // Read input elements into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  // Synchronize (ensure all the data is available)
  __syncthreads();

  // Apply the stencil
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
    result += temp[lindex + offset];
  }

  // Store the result
  out[gindex] = result;
}

void fill_ints(int *x, int n) { fill_n(x, n, 1); }

int main(void) {
  int *in, *out;     // host copies of a, b, c
  int *d_in, *d_out; // device copies of a, b, c

  // Alloc space for host copies and setup values
  static constexpr int PADDED_ARRAY_SIZE_BYTES =
      (PADDED_ARRAY_SIZE) * sizeof(int);
  in = (int *)malloc(PADDED_ARRAY_SIZE_BYTES);
  fill_ints(in, PADDED_ARRAY_SIZE);
  out = (int *)malloc(PADDED_ARRAY_SIZE_BYTES);
  fill_ints(out, PADDED_ARRAY_SIZE);

  // Alloc space for device copies
  cudaMalloc((void **)&d_in, PADDED_ARRAY_SIZE_BYTES);
  cudaMalloc((void **)&d_out, PADDED_ARRAY_SIZE_BYTES);

  // Copy to device
  cudaMemcpy(d_in, in, PADDED_ARRAY_SIZE_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, PADDED_ARRAY_SIZE_BYTES, cudaMemcpyHostToDevice);

  // Launch stencil_1d() kernel on GPU
  // Offset by RADIUS so it starts on the first value and not the padding
  stencil_1d<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS);

  // Copy result back to host
  cudaMemcpy(out, d_out, PADDED_ARRAY_SIZE_BYTES, cudaMemcpyDeviceToHost);

  // Error Checking
  for (int i = 0; i < PADDED_ARRAY_SIZE; ++i) {
    if (i < RADIUS || i >= N + RADIUS) {
      if (out[i] != 1) {
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
      }
    } else {
      if (out[i] != 1 + 2 * RADIUS) {
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i],
               1 + 2 * RADIUS);
      }
    }
  }

  // Cleanup
  free(in);
  free(out);
  cudaFree(d_in);
  cudaFree(d_out);
  printf("Success!\n");
  return 0;
}
