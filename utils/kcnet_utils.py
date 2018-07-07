import os
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

extension_ = load(name='extension_',
                  sources=['utils/wrap.cc', 'utils/batch_knn.cu', 'utils/graph_pooling.cu', 'utils/correlation'],
                  extra_include_paths=['/usr/local/cuda/include', os.path.join(os.getcwd(), 'utils', 'include')],
                  verbose=True)


class BatchKnn(Function):
    """


    :param query: (B, C, N)
           reference: (B, C, M)
           npoints: nearest points numbers
    :return: idxs: (B, N, npoints + 1)
             dists: (B, N, npoints + 1)
    """

    @staticmethod
    def forward(ctx, query, reference, npoints):
        query_t = query.transpose(1, 2).contiguous()
        reference_t = reference.transpose(1, 2).contiguous()
        idxs, dists = extension_.batch_knn_wrapper(query_t, reference_t, npoints + 1)

        return idxs, dists

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None, None


batch_knn = BatchKnn.apply


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


class KernelCorrelation(nn.Module):

    def __init__(self, out_channels, kernel_size):
        super(KernelCorrelation, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, points, knn_graph):
        return x
