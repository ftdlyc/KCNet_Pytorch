#include "cuda_utils.h"
#include "kernel_correlation.h"

// input: points(b, n, npoints, 3), kernel(l, npoints, 3)
// output: outputs(b, l, n)
__global__ void kernel_correlation_kernel(int b, int n, int npoints, int l, int m, float sigma,
                                          const float *__restrict__ points,
                                          const float *__restrict__ kernel,
                                          float *__restrict__ outputs) {
  const int batch_index = blockIdx.x;
  points += batch_index * n * npoints * 3;
  outputs += batch_index * l * n;

  for (int i = threadIdx.y; i < l; i += blockDim.y) {
    const float *temp_k = kernel + i * npoints * 3;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      const float *temp_p = points + j * npoints * 3;

      float sum = 0.;
      for (int k1 = 0; k1 < m; ++k1) {
        float kx = temp_k[k1 * 3];
        float ky = temp_k[k1 * 3 + 1];
        float kz = temp_k[k1 * 3 + 2];
        for (int k2 = 0; k2 < npoints; ++k2) {
          float x = temp_p[k2 * 3];
          float y = temp_p[k2 * 3 + 1];
          float z = temp_p[k2 * 3 + 2];
          float d = (kx - x) * (kx - x) + (ky - y) * (ky - y) + (kz - z) * (kz - z);
          float kc = expf(-1 * d / (2 * sigma * sigma));  // kernel function
          sum += kc;
        }
      }
      outputs[i * n + j] = sum / npoints;
    }
  }
}

// input: grad_outputs(b, l, n), points(b, n, npoints, 3), kernel(l, npoints, 3)
// output: grad_inputs(l, npoints, 3)
__global__ void kernel_correlation_grad_kernel(int b, int n, int npoints, int l, int m, float sigma,
                                               const float *__restrict__ grad_outputs,
                                               const float *__restrict__ points,
                                               const float *__restrict__ kernel,
                                               float *__restrict__ grad_inputs) {
  const int batch_index = blockIdx.x;
  points += batch_index * n * npoints * 3;
  grad_outputs += batch_index * l * n;
  float coef = 1 / (npoints * sigma * sigma);

  for (int i = threadIdx.y; i < l; i += blockDim.y) {
    for (int j = threadIdx.x; j < m; j += blockDim.x) {

      float kx = kernel[(i * m + j) * 3];
      float ky = kernel[(i * m + j) * 3 + 1];
      float kz = kernel[(i * m + j) * 3 + 2];
      float gx = 0.;
      float gy = 0.;
      float gz = 0.;
      for (int k1 = 0; k1 < n; ++k1) {
        const float *temp_p = points + k1 * npoints * 3;

        float sum1 = 0.;
        float sum2 = 0.;
        float sum3 = 0.;
        for (int k2 = 0; k2 < npoints; ++k2) {
          float x = temp_p[k2 * 3];
          float y = temp_p[k2 * 3 + 1];
          float z = temp_p[k2 * 3 + 2];
          float d = (kx - x) * (kx - x) + (ky - y) * (ky - y) + (kz - z) * (kz - z);
          float kc = expf(- d / (2 * sigma * sigma));
          sum1 += (x - kx) * kc;
          sum2 += (y - ky) * kc;
          sum3 += (z - kz) * kc;
        }
         gx += grad_outputs[i * n + k1] * sum1;
         gy += grad_outputs[i * n + k1] * sum2;
         gz += grad_outputs[i * n + k1] * sum3;
      }
      atomicAdd(grad_inputs + (i * m + j) * 3, gx * coef);
      atomicAdd(grad_inputs + (i * m + j) * 3 + 1, gy * coef);
      atomicAdd(grad_inputs + (i * m + j) * 3 + 2, gz * coef);
    }
  }
}

void kernel_correlation_kernel_wrapper(int b, int n, int npoints, int l, int m, float sigma,
                                       const float *points, const float *kernel,
                                       float *outputs) {
  cudaError_t err;
  kernel_correlation_kernel<<<b, opt_block_config(n, l)>>>(b, n, npoints, l, m, sigma, points, kernel, outputs);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void kernel_correlation_grad_kernel_wrapper(int b, int n, int npoints, int l, int m, float sigma,
                                            const float *grad_outputs, const float *points, const float *kernel,
                                            float *grad_inputs) {
  cudaError_t err;
  kernel_correlation_grad_kernel<<<b, opt_block_config(m, l)>>>(
      b, n, npoints, l, m, sigma, grad_outputs, points, kernel, grad_inputs);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
