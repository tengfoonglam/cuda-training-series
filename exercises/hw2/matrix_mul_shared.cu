#include <stdio.h>

// these are just for timing measurments
#include <time.h>

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

static constexpr int kDSize = 8192;
static constexpr int kBlockSize =
    32; // CUDA maximum is 1024 *total* threads in block
static constexpr float kAVal = 3.0f;
static constexpr float kBVal = 2.0f;
static constexpr float kMatrixSizeBytes = kDSize * kDSize * sizeof(float);

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

  // declare cache in shared memory
  __shared__ float As[kBlockSize][kBlockSize];
  __shared__ float Bs[kBlockSize][kBlockSize];

  const int idx =
      threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
  const int idy =
      threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)) {
    float temp = 0.;
    for (int i = 0; i < ds / kBlockSize; ++i) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[idy + i];
      Bs[threadIdx.y][threadIdx.x] = B[idx + i];

      // Synchronize
      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < kBlockSize; ++k) {
        temp += As[i][k] * Bs[k][i]; // dot product of row and column
      }
      __syncthreads();
    }

    // Write to global memory
    C[idy * ds + idx] = temp;
  }
}

int main() {

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;

  // start timing
  t0 = clock();

  h_A = new float[kDSize * kDSize];
  h_B = new float[kDSize * kDSize];
  h_C = new float[kDSize * kDSize];
  for (int i = 0; i < kDSize * kDSize; i++) {
    h_A[i] = kAVal;
    h_B[i] = kBVal;
    h_C[i] = 0;
  }

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, kMatrixSizeBytes);
  cudaMalloc(&d_B, kMatrixSizeBytes);
  cudaMalloc(&d_C, kMatrixSizeBytes);
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, kMatrixSizeBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, kMatrixSizeBytes, cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(kBlockSize, kBlockSize); // dim3 variable holds 3 dimensions
  dim3 grid((kDSize + block.x - 1) / block.x, (kDSize + block.y - 1) / block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, kDSize);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, kMatrixSizeBytes, cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < kDSize * kDSize; i++)
    if (h_C[i] != kAVal * kBVal * kDSize) {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i],
             kAVal * kBVal * kDSize);
      return -1;
    }
  printf("Success!\n");
  return 0;
}
