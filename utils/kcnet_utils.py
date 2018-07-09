import os
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

extension_ = load(name='extension_',
                  sources=['utils/wrap.cc',
                           'utils/batch_knn.cu',
                           'utils/graph_pooling.cu',
                           'utils/group_points.cu',
                           'utils/kernel_correlation.cu'],
                  extra_include_paths=['/usr/local/cuda/include', os.path.join(os.getcwd(), 'utils', 'include')],
                  verbose=False)


class BatchKnn(Function):
    """


    :param query: (B, C, N)
           reference: (B, C, M)
           npoints: nearest points numbers
    :return: idxs: (B, N, npoints)
             dists: (B, N, npoints)
    """

    @staticmethod
    def forward(ctx, query, reference, npoints):
        query_t = query.transpose(1, 2).contiguous()
        reference_t = reference.transpose(1, 2).contiguous()
        idxs, dists = extension_.batch_knn_wrapper(query_t, reference_t, npoints)

        return idxs, dists

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None, None


batch_knn = BatchKnn.apply


class GroupPoints(Function):
    """
    simpling points

    :param features: (B, C, N)
           group_idxs: (B, M, K)
    :return: out: (B, C, M, K)
    """

    @staticmethod
    def forward(ctx, features, group_idxs):
        out = extension_.group_points_wrapper(features, group_idxs)
        ctx.save_for_backward(features, group_idxs)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, group_idxs = ctx.saved_tensors
        n = features.size(2)
        grad_inputs = extension_.group_points_grad_wrapper(grad_out.data.contiguous(), group_idxs, n)

        return grad_inputs, None, None, None, None


group_points = GroupPoints.apply


class GraphPooling(Function):
    """


    :param features: (B, C, N)
           knn_graph: (B, N, npoints)
           weights: (B, N, npoints)
    :return: outputs: (B, C, N)
    """

    @staticmethod
    def forward(ctx, features, knn_graph, weights):
        outputs = extension_.graph_pooling_wrapper(features, knn_graph, weights)
        ctx.save_for_backward(knn_graph, weights)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        knn_graph, weights = ctx.saved_tensors
        grad_inputs = \
            extension_.graph_pooling_grad_wrapper(grad_outputs.contiguous(), knn_graph, weights)
        return grad_inputs, None, None


graph_pooling = GraphPooling.apply


class GraphMaxPooling(Function):
    """


    :param features: (B, C, N)
           knn_graph: (B, N, npoints)
    :return: outputs: (B, C, N)
    """

    @staticmethod
    def forward(ctx, features, knn_graph):
        outputs, idxs = extension_.graph_max_pooling_wrapper(features, knn_graph)
        ctx.save_for_backward(idxs)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        idxs = ctx.saved_tensors[0]
        grad_inputs = extension_.graph_max_pooling_grad_wrapper(grad_outputs.contiguous(), idxs)
        return grad_inputs, None


graph_max_pooling = GraphMaxPooling.apply


class KernelCorrelation(Function):
    """


    :param points: (B, N, npoints, 3)
           kernel: (L, m, 3)
           sigma: gauss kernel sigma
    :return: outputs: (B, L, N)
    """

    @staticmethod
    def forward(ctx, points, kernel, sigma):
        outputs = extension_.kernel_correlation_wrapper(points, kernel, sigma)
        ctx.save_for_backward(points, kernel)
        ctx.sigma = sigma
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        points, kernel = ctx.saved_tensors
        grad_inputs = extension_.kernel_correlation_grad_wrapper(grad_outputs.contiguous(), points, kernel, ctx.sigma)
        return None, grad_inputs, None


kernel_correlation = KernelCorrelation.apply


class LoaclGeometricStructure(nn.Module):
    """


    :param points: (B, 3, N)
           knn_graph: (B, N, npoints)
           sigma: gauss kernel sigma
    :return: out: (B, out_channels, N)
    """

    def __init__(self, out_channels, kernel_size, sigma):
        super(LoaclGeometricStructure, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.kernel = nn.Parameter(torch.Tensor(out_channels, kernel_size, 3))

        self.reset_parameters()

    def reset_parameters(self):
        self.kernel.data.uniform_(-0.2, 0.2)
        return

    def forward(self, points, knn_graph):
        assert (points.size(2) == knn_graph.size(1))
        assert (knn_graph.size(2) == self.kernel_size)
        x = group_points(points, knn_graph)
        x = x - points.unsqueeze(3)
        x = x.transpose(1, 2).transpose(2, 3).contiguous()
        outputs = kernel_correlation(x, self.kernel, self.sigma)
        return outputs
