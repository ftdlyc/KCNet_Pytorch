#include "cuda_utils.h"
#include "graph_pooling.h"

// input: features(b, c, n), knn_graph(b, n, npoints)
// output: outputs(b, c, n), idxs(b, c, n)
__global__ void graph_max_pooling_kernel(int b, int c, int n, int npoints,
                                         const float *__restrict__ features,
                                         const int *__restrict__ knn_graph,
                                         float *__restrict__ outputs,
                                         int *__restrict__ idxs) {
  const int batch_index = blockIdx.x;
  features += batch_index * c * n;
  knn_graph += batch_index * n * npoints;
  outputs += batch_index * c * n;
  idxs += batch_index * c * n;

  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      int besti = -1;
      float best = -1e6;
      for (int k = 0; k < npoints; ++k) {
        int id = knn_graph[j * npoints + k];
        float f = features[i * n + id];
        if(best < f) {
          best = f;
          besti = id;
        }
      }
      outputs[i * n + j] = best;
      idxs[i * n + j] = besti;
    }
  }
}

// input: grad_outputs(b, c, n), idxs(b, c, n)
// output: grad_inputs(b, c, n)
__global__ void graph_max_pooling_grad_kernel(int b, int c, int n,
                                              const float *__restrict__ grad_outputs,
                                              const int *__restrict__ idxs,
                                              float *__restrict__ grad_inputs) {
  const int batch_index = blockIdx.x;
  grad_outputs += batch_index * c * n;
  idxs += batch_index * c * n;
  grad_inputs += batch_index * c * n;

  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      atomicAdd(grad_inputs + i * n + idxs[i * n + j], grad_outputs[i * n + j]);
    }
  }
}

// input: features(b, c, n), knn_graph(b, n, npoints), weights(b, n, npoints)
// output: outputs(b, c, n)
__global__ void graph_pooling_kernel(int b, int c, int n, int npoints,
                                     const float *__restrict__ features,
                                     const int *__restrict__ knn_graph,
                                     const float *__restrict__ weights,
                                     float *__restrict__ outputs) {
  const int batch_index = blockIdx.x;
  features += batch_index * c * n;
  knn_graph += batch_index * n * npoints;
  weights += batch_index * n * npoints;
  outputs += batch_index * c * n;

  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      float f = 0.;
      for (int k = 0; k < npoints; ++k) {
        int id = knn_graph[j * npoints + k];
        f += features[i * n + id] * weights[j * npoints + k];
      }
      outputs[i * n + j] = f;
    }
  }
}

// input: grad_outputs(b, c, n), knn_graph(b, n, npoints), weights(b, n, npoints)
// output: grad_inputs(b, c, n)
__global__ void graph_pooling_grad_kernel(int b, int c, int n, int npoints,
                                          const float *__restrict__ grad_outputs,
                                          const int *__restrict__ knn_graph,
                                          const float *__restrict__ weights,
                                          float *__restrict__ grad_inputs) {
  const int batch_index = blockIdx.x;
  grad_outputs += batch_index * c * n;
  knn_graph += batch_index * n * npoints;
  weights += batch_index * n * npoints;
  grad_inputs += batch_index * c * n;

  for (int i = threadIdx.y; i < c; i += blockDim.y) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      for (int k = 0; k < npoints; ++k) {
        atomicAdd(grad_inputs + i * n + knn_graph[j * npoints + k],
                  grad_outputs[i * n + j] * weights[j * npoints + k]);
      }
    }
  }
}

void graph_max_pooling_kernel_wrapper(int b, int c, int n, int npoints,
                                      const float *features, const int *knn_graph,
                                      float *outputs, int *idxs) {
  cudaError_t err;
  graph_max_pooling_kernel<<<b, opt_block_config(n, c)>>>(b, c, n, npoints, features, knn_graph, outputs, idxs);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void graph_max_pooling_grad_kernel_wrapper(int b, int c, int n,
                                           const float *grad_outputs, const int *idxs,
                                           float *grad_inputs) {
  cudaError_t err;
  graph_max_pooling_grad_kernel<<<b, opt_block_config(n, c)>>>(b, c, n, grad_outputs, idxs, grad_inputs);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void graph_pooling_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *features, const int *knn_graph, const float *weights,
                                  float *outputs) {
  cudaError_t err;
  graph_pooling_kernel<<<b, opt_block_config(n, c)>>>(b, c, n, npoints, features, knn_graph, weights, outputs);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void graph_pooling_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_outputs, const int *knn_graph, const float *weights,
                                       float *grad_inputs) {
  cudaError_t err;
  graph_pooling_grad_kernel<<<b, opt_block_config(n, c)>>>(b, c, n, npoints, grad_outputs, knn_graph, weights, grad_inputs);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
