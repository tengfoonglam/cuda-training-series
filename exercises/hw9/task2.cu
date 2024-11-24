#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>

typedef int mytype;
const int test_dsize = 256;
const int nTPB = 256;
static constexpr size_t kSharedMemUsed{0};

template <typename T> __device__ size_t predicate_test(T data, T testval) {
  if (data == testval)
    return 0;
  return 1;
}

using namespace cooperative_groups;

// assume dsize is divisbile by nTPB
template <typename T>
__global__ void my_remove_if(const T *__restrict__ idata, const T remove_val,
                             T *__restrict__ odata, size_t *__restrict__ idxs,
                             const size_t dsize) {

  __shared__ size_t sidxs[nTPB];
  auto g = this_thread_block();
  auto gg = this_grid();
  size_t tidx = g.thread_rank();
  size_t gidx = tidx + nTPB * g.group_index().x;
  size_t gridSize = g.size() * gridDim.x;

  // first use grid-stride loop to have each block do a prefix sum over data set
  // Note: operation is done per block so we sync block-level
  for (size_t i = gidx; i < dsize; i += gridSize) {
    size_t temp = predicate_test(idata[i], remove_val);
    sidxs[tidx] = temp;
    for (size_t j = 1; j < g.size(); j <<= 1) {
      g.sync();
      if (j <= tidx) {
        temp += sidxs[tidx - j];
      }
      g.sync();
      if (j <= tidx) {
        sidxs[tidx] = temp;
      }
    }
    idxs[i] = temp;
    g.sync();
  }

  // grid-wide barrier (sync all blocks)
  gg.sync();

  // then compute final index, and move input data to output location
  size_t stride = 0;
  for (size_t i = gidx; i < dsize; i += gridSize) {
    T temp = idata[i];
    if (predicate_test(temp, remove_val)) {
      size_t my_idx = idxs[i];
      for (size_t j = 1; (j - 1) < (g.group_index().x + (stride * gridDim.x));
           ++j) {
        my_idx += idxs[j * nTPB - 1];
      }
      odata[my_idx - 1] = temp;
    }
    ++stride;
  }
}

int main() {
  // data setup
  mytype *d_idata, *d_odata, *h_data;
  size_t *d_idxs;
  size_t tsize = ((size_t)test_dsize) * sizeof(mytype);
  h_data = (mytype *)malloc(tsize);
  cudaMalloc(&d_idata, tsize);
  cudaMalloc(&d_odata, tsize);
  cudaMemset(d_odata, 0, tsize);
  cudaMalloc(&d_idxs, test_dsize * sizeof(size_t));
  // check for support and device configuration
  // and calculate maximum grid size
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    printf("cuda error: %s\n", cudaGetErrorString(err));
    return 0;
  }
  if (prop.cooperativeLaunch == 0) {
    printf("cooperative launch not supported\n");
    return 0;
  }
  int numSM = prop.multiProcessorCount;
  printf("number of SMs = %u\n", numSM);
  int numBlkPerSM;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlkPerSM,
                                                my_remove_if<mytype>, nTPB, 0);
  printf("number of blocks per SM = %u\n", numBlkPerSM);
  // test 1: no remove values
  for (size_t i = 0; i < test_dsize; ++i) {
    h_data[i] = i;
  }

  cudaMemcpy(d_idata, h_data, tsize, cudaMemcpyHostToDevice);
  cudaStream_t str;
  cudaStreamCreate(&str);
  mytype remove_val = -1;
  size_t ds = test_dsize;
  void *args[] = {(void *)&d_idata, (void *)&remove_val, (void *)&d_odata,
                  (void *)&d_idxs, (void *)&ds};
  dim3 grid(numBlkPerSM * numSM);
  dim3 block(nTPB);
  cudaLaunchCooperativeKernel((void *)my_remove_if<mytype>, grid, block, args,
                              kSharedMemUsed, str);
  err = cudaMemcpy(h_data, d_odata, tsize, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printf("cuda error: %s\n", cudaGetErrorString(err));
    return 0;
  }
  // validate
  for (size_t i = 0; i < test_dsize; ++i) {
    if (h_data[i] != i) {
      printf("mismatch 1 at %zu, was: %d, should be: %zu\n", i, h_data[i], i);
      return 1;
    }
  }

  printf("No remove values test case succeeded");
  // test 2: with remove values
  int val = 0;
  for (size_t i = 0; i < test_dsize; ++i) {
    h_data[i] = ((rand() / (float)RAND_MAX) > 0.5) ? val++ : -1;
  }
  thrust::device_vector<mytype> t_data(h_data, h_data + test_dsize);
  cudaMemcpy(d_idata, h_data, tsize, cudaMemcpyHostToDevice);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaLaunchCooperativeKernel((void *)my_remove_if<mytype>, grid, block, args,
                              kSharedMemUsed, str);
  cudaEventRecord(stop);
  float et;
  cudaMemcpy(h_data, d_odata, tsize, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&et, start, stop);
  // validate
  for (size_t i = 0; i < val; ++i) {
    if (h_data[i] != i) {
      printf("mismatch 2 at %zu, was: %d, should be: %zu\n", i, h_data[i], i);
      return 1;
    }
  }

  printf("kernel time: %fms\n", et);
  cudaEventRecord(start);
  thrust::remove(t_data.begin(), t_data.end(), -1);
  cudaEventRecord(stop);
  thrust::host_vector<mytype> th_data = t_data;
  // validate
  for (size_t i = 0; i < val; ++i) {
    if (h_data[i] != th_data[i]) {
      printf("mismatch 3 at %zu, was: %d, should be: %d\n", i, th_data[i],
             h_data[i]);
      return 1;
    }
  }

  cudaEventElapsedTime(&et, start, stop);
  printf("thrust time: %fms\n", et);
  printf("Remove values test case succeeded");
  return 0;
}
