
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor furthest_point_sampling(torch::Tensor points, const int nsamples, const int keepsamples);
