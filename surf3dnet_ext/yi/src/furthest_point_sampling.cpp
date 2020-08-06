
#include "furthest_point_sampling.h"
#include "furthest_point_sampling_kernel.h"

torch::Tensor furthest_point_sampling(torch::Tensor points, const int nsamples, const int keepsamples)
{
    TORCH_CHECK(points.type().is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "points must be a contiguous tensor");
    TORCH_CHECK(points.scalar_type() == torch::ScalarType::Float, "points must be a float tensor");
    TORCH_CHECK(points.dim() == 3 && (points.size(2) == 3 || points.size(2) == 4), "points must be a [B][N][3] or [B][N][4] tensor");

    torch::Tensor output = torch::zeros({ points.size(0), nsamples }, torch::device(points.device()).dtype(torch::ScalarType::Int));
    torch::Tensor tmp = torch::full({ points.size(0), points.size(1) }, 1e10, torch::device(points.device()).dtype(torch::ScalarType::Float));

    int cudadev = points.device().index();

#pragma omp parallel for
    for (int i = 0; i < 1; i++)
    {
        cudaSetDevice(cudadev);
        furthest_point_sampling_kernel_wrapper(
            points.size(0), points.size(1), nsamples, points.size(2), keepsamples, points.data_ptr<float>(),
            tmp.data_ptr<float>(), output.data_ptr<int>());
    }

    return output;
}
