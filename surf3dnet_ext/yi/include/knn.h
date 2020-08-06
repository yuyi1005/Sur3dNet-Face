
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

std::vector<torch::Tensor> knn(torch::Tensor models, torch::Tensor points, const int k);
