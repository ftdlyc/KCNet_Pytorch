#ifndef KERNEL_CORRELATION_H
#define KERNEL_CORRELATION_H

void kernel_correlation_kernel_wrapper(int b, int n, int npoints, int l, int m, float sigma,
                                       const float *points, const float *kernel,
                                       float *outputs);

void kernel_correlation_grad_kernel_wrapper(int b, int n, int npoints, int l, int m, float sigma,
                                            const float *grad_outputs, const float *points, const float *kernel,
                                            float *grad_inputs);

#endif