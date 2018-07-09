#include "cuda_utils.h"
#include "group_points.h"

__global__ void group_points_kernel(int b, int c, int n, int m, int k,
                                    const float *__restrict__ points,
                                    const int *__restrict__ group_idxs,
                                    float *__restrict__ out) {
  const int batch_index = blockIdx.x;
  points += batch_index * c * n;
  group_idxs += batch_index * m * k;
  out += batch_index * c * m * k;

  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      for (int l = 0; l < k; ++l) {
        out[(i * m + j) * k + l] = points[i * n + group_idxs[j * k + l]];
      }
    }
  }
}

// input: points(b, c, n) group_idxs(b, m, k)
// output: out(b, c, m, k)
void group_points_kernel_wrapper(int b, int c, int n, int m, int k,
                                 const float *points, const int *group_idxs,
                                 float *out) {
  cudaError_t err;
  group_points_kernel<<<b, opt_block_config(m, c)>>>(
      b, c, n, m, k, points, group_idxs, out);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

__global__ void group_points_grad_kernel(int b, int c, int n, int m, int k,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ group_idxs, 
                                         float *__restrict__ grad_points) {
  const int batch_index = blockIdx.x;
  grad_out += batch_index * c * m * k;
  group_idxs += batch_index * m * k;
  grad_points += batch_index * c * n;

  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      for (int l = 0; l < k; ++l) {
        atomicAdd(grad_points + i * n + group_idxs[j * k + l], grad_out[(i * m + j) * k + l]);
      }
    }
  }
}

// input: grad_out(b, c, m, k), group_idxs(b, m, k)
// output: grad_points(b, c, n)
void group_points_grad_kernel_wrapper(int b, int c, int n, int m, int k,
                                      const float *grad_out,
                                      const int *group_idxs,
                                      float *grad_points) {
  cudaError_t err;
  group_points_grad_kernel<<<b, opt_block_config(m, c)>>>(
      b, c, n, m, k, grad_out, group_idxs, grad_points);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
