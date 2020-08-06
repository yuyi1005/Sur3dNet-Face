
#include <stdio.h>
#include <stdlib.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#define TOTAL_THREADS 512

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, s), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, int s, int keep, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
  if (m <= 0) return;
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * s;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (tid == 0)
  {
    if (s == 4)
    {
      float maxw = 0;
      for (int k = 0; k < n; k++)
      {
        float w2 = dataset[k * s + 3];
        if (maxw < w2)
        {
          maxw = w2;
          old = k;
        }
      }
    }
    idxs[0] = old;
  }

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * s + 0];
    float y1 = dataset[old * s + 1];
    float z1 = dataset[old * s + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2, w2;
      x2 = dataset[k * s + 0];
      y2 = dataset[k * s + 1];
      z2 = dataset[k * s + 2];
      if (s == 4)
        w2 = dataset[k * s + 3];
      else
        w2 = 1;

      float d =
          ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)) * w2;

      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

	if (j < keep)
    {
	  old = j;
    }
	else
    {
      if (block_size >= 512) {
        if (tid < 256) {
          __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
      }
      if (block_size >= 256) {
        if (tid < 128) {
          __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
      }
      if (block_size >= 128) {
        if (tid < 64) {
          __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
      }
      if (block_size >= 64) {
        if (tid < 32) {
          __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
      }
      if (block_size >= 32) {
        if (tid < 16) {
          __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
      }
      if (block_size >= 16) {
        if (tid < 8) {
          __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
      }
      if (block_size >= 8) {
        if (tid < 4) {
          __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
      }
      if (block_size >= 4) {
        if (tid < 2) {
          __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
      }
      if (block_size >= 2) {
        if (tid < 1) {
          __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
      }

      old = dists_i[0];
    }

    if (tid == 0) idxs[j] = old;
  }
}

void furthest_point_sampling_kernel_wrapper(int b, int n, int m, int s, int keep, 
                                            const float *dataset, float *temp,
                                            int *idxs) {
  unsigned int n_threads = opt_n_threads(n);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
    case 512:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_kernel<256>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_kernel<128>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_kernel<64>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_kernel<32>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_kernel<16>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_kernel<8>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_kernel<4>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_kernel<2>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_kernel<1>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, s, keep, dataset, temp, idxs);
  }
}
