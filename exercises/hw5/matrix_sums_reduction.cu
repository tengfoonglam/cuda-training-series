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

const size_t DSIZE = 16384; // matrix side dimension
const int block_size = 256; // CUDA maximum is 1024
// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds) {

  int idx =
      threadIdx.x +
      blockDim.x *
          blockIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds) {
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum +=
          A[idx * ds +
            i]; // write a for loop that will cause the thread to iterate across
                // a row, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
  }
}

__global__ void row_sums_reduction(const float *A, float *sums, size_t ds) {
  // Shared memory: number of threads
  __shared__ float sdata[block_size];
  const size_t row_idx = blockIdx.x; // each block handles a row
  const size_t index_at_start_of_row = row_idx * ds;
  const size_t tid = threadIdx.x;
  sdata[tid] = 0.0f;
  size_t i = threadIdx.x;

  if (row_idx >= ds) {
    return;
  }

  // block stride sum to shared memory
  while (i < ds) { // grid stride loop to load data
    sdata[tid] += A[index_at_start_of_row + i];
    i += blockDim.x;
  }

  // sum shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
  }
  if (tid == 0) {
    sums[row_idx] = sdata[0];
  }
}

// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds) {

  int idx =
      threadIdx.x +
      blockDim.x *
          blockIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds) {
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[idx + ds * i]; // write a for loop that will cause the thread to
                              // iterate down a column, keeeping a running sum,
                              // and write the result to sums
    sums[idx] = sum;
  }
}
bool validate(float *data, size_t sz) {
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {
      printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i],
             (float)sz);
      return false;
    }
  return true;
}

void run_row_sum_naive() {
  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE * DSIZE]; // allocate space for data in host memory
  h_sums = new float[DSIZE]();
  for (int i = 0; i < DSIZE * DSIZE; ++i) // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A,
             DSIZE * DSIZE * sizeof(float)); // allocate device space for A
  cudaMalloc(&d_sums,
             DSIZE * sizeof(float)); // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  // cuda processing sequence step 1 is complete
  row_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(d_A, d_sums,
                                                                  DSIZE);
  cudaCheckErrors("kernel launch failure");
  // cuda processing sequence step 2 is complete
  //  copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  // cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE))
    printf("row sums incorrect!\n");
  else
    printf("row sums correct!\n");
}

void run_row_sum_reduction() {

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE * DSIZE]; // allocate space for data in host memory
  h_sums = new float[DSIZE]();
  for (int i = 0; i < DSIZE * DSIZE; ++i) // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A,
             DSIZE * DSIZE * sizeof(float)); // allocate device space for A
  cudaMalloc(&d_sums,
             DSIZE * sizeof(float)); // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  // One block per row
  row_sums_reduction<<<DSIZE, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE))
    printf("row sums reduction incorrect!\n");
  else
    printf("row sums reduction correct!\n");
}

void run_col_sum() {
  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE * DSIZE]; // allocate space for data in host memory
  h_sums = new float[DSIZE]();
  for (int i = 0; i < DSIZE * DSIZE; ++i) // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A,
             DSIZE * DSIZE * sizeof(float)); // allocate device space for A
  cudaMalloc(&d_sums,
             DSIZE * sizeof(float)); // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  // cuda processing sequence step 1 is complete
  column_sums<<<(DSIZE + block_size - 1) / block_size, block_size>>>(
      d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  // cuda processing sequence step 2 is complete
  //  copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
  // cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE))
    printf("column sums incorrect!\n");
  else
    printf("column sums correct!\n");
}

int main() {
  run_row_sum_naive();
  run_row_sum_reduction();
  run_col_sum();
  return 0;
}
