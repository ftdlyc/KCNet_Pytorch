#include "cuda_utils.h"
#include "batch_knn.h"

// input: query(b, n, c), reference(b, m, c)
// output: idxs(b, n, npoints), dists(b, n, npoints)
__global__ void batch_knn_kernel(int b, int n, int c, int m, int npoints,
                                 const float *__restrict__ query,
                                 const float *__restrict__ reference,
                                 int *__restrict__ idxs,
                                 float *__restrict__ dists,
                                 float *__restrict__ temp) {
  const int batch_index = blockIdx.x;
  query += batch_index * n * c;
  reference += batch_index * m * c;
  idxs += batch_index * n * npoints;
  dists += batch_index * n * npoints;
  temp += batch_index * n * m;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float *q = query + i * c;
    float *d = temp + i * m;
    for (int j = 0; j < m; ++j) {
      const float *r = reference + j * 3;
      d[j] = 0.;
      for (int k = 0; k < c; ++k) {
        d[j] += (q[k] - r[k]) * (q[k] - r[k]);
      }
    }
    for (int k = 0; k < npoints; ++k) {
      int besti = -1;
      float best_dist = 1e5;
      for (int j = 0; j < m; ++j) {
        if(d[j] < best_dist) {
          besti = j;
          best_dist = d[j];
        }
      }
      d[besti] = 1e6;
      idxs[i * npoints + k] = besti;
      dists[i * npoints + k] = best_dist;
    }
  }
}

void batch_knn_kernel_wrapper(int b, int n, int c, int m, int npoints, const float *query,
                              const float *reference, int *idxs, float *dists, float *temp) {
  cudaError_t err;
  batch_knn_kernel<<<b, opt_n_threads(n)>>>(b, n, c, m, npoints, query, reference, idxs, dists, temp);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}