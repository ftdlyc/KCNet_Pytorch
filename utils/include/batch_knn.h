#ifndef BATCH_KNN_H
#define BATCH_KNN_H

void batch_knn_kernel_wrapper(int b, int n, int c, int m, int npoints, const float *query,
                              const float *reference, int *idxs, float *dists, float *temp);

#endif