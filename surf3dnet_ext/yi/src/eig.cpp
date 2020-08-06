
#include "eig.h"
#include "eig_kernel.h"

std::vector<torch::Tensor> eig(torch::Tensor x)
{
    TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be a contiguous tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float, "x must be a float tensor");
    TORCH_CHECK(x.dim() == 3 && ((x.size(1) == 3 && x.size(2) == 3) || (x.size(1) == 4 && x.size(2) == 4)), "x must be a [B][3][3] or [B][4][4] tensor");

    int b = x.size(0);
    int d = x.size(1);

    torch::Tensor val = torch::zeros({ b, d, 1 }, torch::device(x.device()).dtype(torch::ScalarType::Float));
    torch::Tensor vec = torch::zeros({ b, d, d }, torch::device(x.device()).dtype(torch::ScalarType::Float));

    int cudadev = x.device().index();

#pragma omp parallel for
    for (int i = 0; i < 1; i++)
    {
        cudaSetDevice(cudadev);
        calc_eig(x.data_ptr<float>(), val.data_ptr<float>(), vec.data_ptr<float>(), b, d);
    }

    return { val, vec };
}
