#include <stdio.h>

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

static constexpr int DSIZE = 4096;
static constexpr int BLOCK_SIZE = 256; // CUDA maximum is 1024
// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds) {
  if (const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
      idx < ds) // create typical 1D thread index from built-in variables
  {
    C[idx] = A[idx] + B[idx]; // do the vector (element) add here
  }
}

int main() {

  static constexpr auto FLOAT_SIZE = sizeof(float);
  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE]; // allocate space for vectors in host memory
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  for (std::size_t i = 0; i < DSIZE; ++i) { // initialize vectors in host memory
    h_A[i] = rand() / static_cast<float>(RAND_MAX);
    h_B[i] = rand() / static_cast<float>(RAND_MAX);
    h_C[i] = 0.;
  }
  cudaMalloc(&d_A, DSIZE * FLOAT_SIZE);  // allocate device space for vector A
  cudaMalloc(&d_B, DSIZE * FLOAT_SIZE);  // allocate device space for vector B
  cudaMalloc(&d_C, DSIZE * FLOAT_SIZE);  // allocate device space for vector C
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy vector A to device:
  cudaMemcpy(d_A, h_A, DSIZE * FLOAT_SIZE, cudaMemcpyHostToDevice);
  // copy vector B to device:
  cudaMemcpy(d_B, h_B, DSIZE * FLOAT_SIZE, cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  // cuda processing sequence step 1 is complete
  vadd<<<(DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C,
                                                              DSIZE);
  cudaCheckErrors("kernel launch failure");
  // cuda processing sequence step 2 is complete
  //  copy vector C from device to host:
  cudaMemcpy(h_C, d_C, DSIZE * FLOAT_SIZE, cudaMemcpyDeviceToHost);
  // cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  printf("A[0] = %f\n", h_A[0]);
  printf("B[0] = %f\n", h_B[0]);
  printf("C[0] = %f\n", h_C[0]);
  return 0;
}
