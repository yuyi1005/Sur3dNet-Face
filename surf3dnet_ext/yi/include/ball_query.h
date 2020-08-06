
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor ball_query(torch::Tensor models, torch::Tensor points, const float radius,
                      const int nsample);
