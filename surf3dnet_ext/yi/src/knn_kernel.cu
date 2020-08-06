
#include "knn_kernel.h"
#include <device_launch_parameters.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <flann/flann.hpp>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

//models[M][3] points[N][3] neighb[N][K] distan[N][K]
void calc_knn(float *models, float *points, int *neighb, float *distan, int M, int N, int K)
{
    flann::Matrix<float> flann_mat_model((float *)models, M, 3);

    flann::KDTreeCuda3dIndexParams params(M > 32 ? 32 : M);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    params["gpu_stream"] = stream;
    flann::Index<flann::L2<float>> flannindex_var(flann_mat_model, params);
    flann::Index<flann::L2<float>> *flannindex = &flannindex_var;

    flannindex->buildIndex();

    flann::Matrix<float> flann_mat_p(points, N, 3);
    flann::Matrix<int> flann_mat_idx(neighb, N, K);
    flann::Matrix<float> flann_mat_dis(distan, N, K);

    flann::SearchParams parasearch;
    parasearch.matrices_in_gpu_ram = true;
    flannindex->knnSearch(flann_mat_p, flann_mat_idx, flann_mat_dis, K, parasearch);
}
