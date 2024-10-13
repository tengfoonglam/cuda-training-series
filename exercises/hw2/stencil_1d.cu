#include <algorithm>
#include <stdio.h>

using namespace std;

namespace {

static constexpr int kN = 4096;
static constexpr int kRadius = 3;
static constexpr int kBlockSize = 16;
static constexpr int kPaddedArraySize = 2 * kRadius + kN;

__global__ void stencil_1d(int *in, int *out) {
  __shared__ int temp[kBlockSize + 2 * kRadius];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;

  // Index of current input value (with gindex) in temp store
  // kNote temp is padded on both sides with kRadius
  int lindex = threadIdx.x + kRadius;

  // Read input elements into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < kRadius) {
    temp[lindex - kRadius] = in[gindex - kRadius];
    temp[lindex + kBlockSize] = in[gindex + kBlockSize];
  }

  // Synchronize (ensure all the data is available)
  __syncthreads();

  // Apply the stencil
  int result = 0;
  for (int offset = -kRadius; offset <= kRadius; ++offset) {
    result += temp[lindex + offset];
  }

  // Store the result
  out[gindex] = result;
}

void fill_ints(int *x, int n) { fill_n(x, n, 1); }

} // namespace

int main(void) {
  int *in, *out;     // host copies of a, b, c
  int *d_in, *d_out; // device copies of a, b, c

  // Alloc space for host copies and setup values
  static constexpr int kPaddedArraySize_BYTES =
      (kPaddedArraySize) * sizeof(int);
  in = (int *)malloc(kPaddedArraySize_BYTES);
  fill_ints(in, kPaddedArraySize);
  out = (int *)malloc(kPaddedArraySize_BYTES);
  fill_ints(out, kPaddedArraySize);

  // Alloc space for device copies
  cudaMalloc((void **)&d_in, kPaddedArraySize_BYTES);
  cudaMalloc((void **)&d_out, kPaddedArraySize_BYTES);

  // Copy to device
  cudaMemcpy(d_in, in, kPaddedArraySize_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, kPaddedArraySize_BYTES, cudaMemcpyHostToDevice);

  // Launch stencil_1d() kernel on GPU
  // Offset by kRadius so it starts on the first value and not the padding
  stencil_1d<<<kN / kBlockSize, kBlockSize>>>(d_in + kRadius, d_out + kRadius);

  // Copy result back to host
  cudaMemcpy(out, d_out, kPaddedArraySize_BYTES, cudaMemcpyDeviceToHost);

  // Error Checking
  for (int i = 0; i < kPaddedArraySize; ++i) {
    if (i < kRadius || i >= kN + kRadius) {
      if (out[i] != 1) {
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
      }
    } else {
      if (out[i] != 1 + 2 * kRadius) {
        printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i],
               1 + 2 * kRadius);
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
