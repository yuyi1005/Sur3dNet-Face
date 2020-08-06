
#include "ball_query.h"
#include "ball_query_kernel.h"

torch::Tensor ball_query(torch::Tensor models, torch::Tensor points, const float radius, const int nsample)
{
    TORCH_CHECK(models.type().is_cuda(), "models must be a CUDA tensor");
    TORCH_CHECK(models.is_contiguous(), "models must be a contiguous tensor");
    TORCH_CHECK(models.scalar_type() == torch::ScalarType::Float, "models must be a float tensor");
    TORCH_CHECK(models.dim() == 3 && models.size(2) == 3, "models must be a [B][M][3] tensor");

    TORCH_CHECK(points.type().is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "points must be a contiguous tensor");
    TORCH_CHECK(points.scalar_type() == torch::ScalarType::Float, "points must be a float tensor");
    TORCH_CHECK(points.dim() == 3 && points.size(2) == 3, "points must be a [B][N][3] tensor");

    torch::Tensor idx = torch::zeros({ points.size(0), points.size(1), nsample }, torch::device(points.device()).dtype(torch::ScalarType::Int));

    int cudadev = points.device().index();

#pragma omp parallel for
    for (int i = 0; i < 1; i++)
    {
        cudaSetDevice(cudadev);
        query_ball_point_kernel_wrapper(models.size(0), models.size(1), points.size(1),
            radius, nsample, points.data_ptr<float>(),
            models.data_ptr<float>(), idx.data_ptr<int>());
    }

    return idx;
}
