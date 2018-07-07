#include <vector>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "batch_knn.h"
#include "graph_pooling.h"

std::vector<at::Tensor> batch_knn_wrapper(at::Tensor query_tensor, at::Tensor reference_tensor, int npoints) {
  CHECK_INPUT(query_tensor);
  CHECK_INPUT_TYPE(query_tensor, at::ScalarType::Float);
  CHECK_INPUT(reference_tensor);
  CHECK_INPUT_TYPE(reference_tensor, at::ScalarType::Float);
  int b = query_tensor.size(0);
  int n = query_tensor.size(1);
  int c = query_tensor.size(2);
  int m = reference_tensor.size(1);
  AT_ASSERT(m > npoints, "npoints must smaller than m")
  at::Tensor idxs_tensor = at::zeros(torch::CUDA(at::kInt), {b, n, npoints});
  at::Tensor dists_tensor = at::zeros(torch::CUDA(at::kFloat), {b, n, npoints});
  at::Tensor temp_tensor = at::zeros(torch::CUDA(at::kFloat), {b, n, m});
  const float *query = query_tensor.data<float>();
  const float *reference = reference_tensor.data<float>();
  int *idxs = idxs_tensor.data<int>();
  float *dists = dists_tensor.data<float>();
  float *temp = temp_tensor.data<float>();

  batch_knn_kernel_wrapper(b, n, c, m, npoints, query, reference, idxs, dists, temp);
  return {idxs_tensor, dists_tensor};
}

std::vector<at::Tensor> graph_max_pooling_wrapper(at::Tensor features_tensor, at::Tensor knn_graph_tensor) {
  CHECK_INPUT(features_tensor);
  CHECK_INPUT_TYPE(features_tensor, at::ScalarType::Float);
  CHECK_INPUT(knn_graph_tensor);
  CHECK_INPUT_TYPE(knn_graph_tensor, at::ScalarType::Int);
  int b = features_tensor.size(0);
  int c = features_tensor.size(1);
  int n = features_tensor.size(2);
  int npoints = knn_graph_tensor.size(2);
  at::Tensor outputs_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, n});
  at::Tensor idxs_tensor = at::zeros(torch::CUDA(at::kInt), {b, c, n});
  const float *features = features_tensor.data<float>();
  const int *knn_graph = knn_graph_tensor.data<int>();
  float *outputs = outputs_tensor.data<float>();
  int *idxs = idxs_tensor.data<int>();

  graph_max_pooling_kernel_wrapper(b, c, n, npoints, features, knn_graph, outputs, idxs);
  return {outputs_tensor, idxs_tensor};
}

at::Tensor graph_max_pooling_grad_wrapper(at::Tensor grad_outputs_tensor, at::Tensor idxs_tensor) {
  CHECK_INPUT(grad_outputs_tensor);
  CHECK_INPUT_TYPE(grad_outputs_tensor, at::ScalarType::Float);
  CHECK_INPUT(idxs_tensor);
  CHECK_INPUT_TYPE(idxs_tensor, at::ScalarType::Int);
  int b = grad_outputs_tensor.size(0);
  int c = grad_outputs_tensor.size(1);
  int n = grad_outputs_tensor.size(2);
  at::Tensor grad_inputs_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, n});
  const float *grad_outputs = grad_outputs_tensor.data<float>();
  const int *idxs = idxs_tensor.data<int>();
  float *grad_inputs = grad_inputs_tensor.data<float>();

  graph_max_pooling_grad_kernel_wrapper(b, c, n, grad_outputs, idxs, grad_inputs);
  return grad_inputs_tensor;
}

at::Tensor graph_pooling_wrapper(at::Tensor features_tensor,
                                 at::Tensor knn_graph_tensor,
                                 at::Tensor weights_tensor) {
  CHECK_INPUT(features_tensor);
  CHECK_INPUT_TYPE(features_tensor, at::ScalarType::Float);
  CHECK_INPUT(knn_graph_tensor);
  CHECK_INPUT_TYPE(knn_graph_tensor, at::ScalarType::Int);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);
  int b = features_tensor.size(0);
  int c = features_tensor.size(1);
  int n = features_tensor.size(2);
  int npoints = knn_graph_tensor.size(2);
  at::Tensor outputs_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, n});
  const float *features = features_tensor.data<float>();
  const int *knn_graph = knn_graph_tensor.data<int>();
  const float *weights = weights_tensor.data<float>();
  float *outputs = outputs_tensor.data<float>();

  graph_pooling_kernel_wrapper(b, c, n, npoints, features, knn_graph, weights, outputs);
  return outputs_tensor;
}

at::Tensor graph_pooling_grad_wrapper(at::Tensor grad_outputs_tensor,
                                      at::Tensor knn_graph_tensor,
                                      at::Tensor weights_tensor) {
  CHECK_INPUT(grad_outputs_tensor);
  CHECK_INPUT_TYPE(grad_outputs_tensor, at::ScalarType::Float);
  CHECK_INPUT(knn_graph_tensor);
  CHECK_INPUT_TYPE(knn_graph_tensor, at::ScalarType::Int);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);
  int b = grad_outputs_tensor.size(0);
  int c = grad_outputs_tensor.size(1);
  int n = grad_outputs_tensor.size(2);
  int npoints = knn_graph_tensor.size(2);
  at::Tensor grad_inputs_tensor = at::zeros(torch::CUDA(at::kFloat), {b, c, n});
  const float *grad_outputs = grad_outputs_tensor.data<float>();
  const int *knn_graph = knn_graph_tensor.data<int>();
  const float *weights = weights_tensor.data<float>();
  float *grad_inputs = grad_inputs_tensor.data<float>();

  graph_pooling_grad_kernel_wrapper(b, c, n, npoints, grad_outputs, knn_graph, weights, grad_inputs);
  return grad_inputs_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_knn_wrapper", &batch_knn_wrapper, "batch knn");
    m.def("graph_max_pooling_wrapper", &graph_max_pooling_wrapper, "graph max pooling");
    m.def("graph_max_pooling_grad_wrapper", &graph_max_pooling_grad_wrapper, "graph max pooling grad");
    m.def("graph_pooling_wrapper", &graph_pooling_wrapper, "graph pooling");
    m.def("graph_pooling_grad_wrapper", &graph_pooling_grad_wrapper, "graph pooling grad");
}
