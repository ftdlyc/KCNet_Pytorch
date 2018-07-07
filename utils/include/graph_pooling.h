#ifndef GRAPH_POOLING
#define GRAPH_POOLING

void graph_max_pooling_kernel_wrapper(int b, int c, int n, int npoints,
                                      const float *features, const int *knn_graph,
                                      float *outputs, int *idxs);

void graph_max_pooling_grad_kernel_wrapper(int b, int c, int n,
                                           const float *grad_outputs, const int *idxs,
                                           float *grad_inputs);

void graph_pooling_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *features, const int *knn_graph, const float *weights,
                                  float *outputs);

void graph_pooling_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_outputs, const int *knn_graph, const float *weights,
                                       float *grad_inputs);

#endif