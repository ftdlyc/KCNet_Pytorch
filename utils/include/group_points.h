#ifndef GROUP_POINTS_H
#define GROUP_POINTS_H

void group_points_kernel_wrapper(int b, int c, int n, int m, int k,
                                 const float *points, const int *group_idxs,
                                 float *out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int m, int k,
                                      const float *grad_out, const int *group_idxs,
                                      float *grad_points);

#endif