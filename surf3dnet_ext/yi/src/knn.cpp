
#include "knn.h"
#include "knn_kernel.h"

std::vector<torch::Tensor> knn(torch::Tensor models, torch::Tensor points, const int k)
{
    TORCH_CHECK(!models.type().is_cuda(), "models must be a CPU tensor");
    TORCH_CHECK(models.is_contiguous(), "models must be a contiguous tensor");
    TORCH_CHECK(models.scalar_type() == torch::ScalarType::Float, "models must be a float tensor");
    TORCH_CHECK(models.dim() == 3 && models.size(2) == 3, "models must be a [B][M][3] tensor");

    TORCH_CHECK(points.type().is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "points must be a contiguous tensor");
    TORCH_CHECK(points.scalar_type() == torch::ScalarType::Float, "points must be a float tensor");
    TORCH_CHECK(points.dim() == 3 && points.size(2) == 3, "points must be a [B][N][3] tensor");

    TORCH_CHECK(models.size(0) == points.size(0), "models and points should have same batch size");
    TORCH_CHECK(k > 0 && k <= models.size(1), "k must be a positive int number not grater than models.size(1)");

    int b = models.size(0);
    int m = models.size(1);
    int n = points.size(1);

    torch::Tensor neighb = torch::zeros({ b, n, k }, torch::device(points.device()).dtype(torch::ScalarType::Int));
    torch::Tensor distan = torch::zeros({ b, n, k }, torch::device(points.device()).dtype(torch::ScalarType::Float));

    int cudadev = points.device().index();

#pragma omp parallel for
    for (int i = 0; i < b; i++)
    {
        cudaSetDevice(cudadev);
        calc_knn(models.data_ptr<float>() + i * m * 3, points.data_ptr<float>() + i * n * 3,
            neighb.data_ptr<int>() + i * n * k, distan.data_ptr<float>() + i * n * k, m, n, k);
    }

    return { distan, neighb };
}
