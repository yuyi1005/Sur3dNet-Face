
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

std::vector<torch::Tensor> eig(torch::Tensor x);
