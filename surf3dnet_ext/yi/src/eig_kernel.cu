
#include "eig_kernel.h"
#include <device_launch_parameters.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

__device__ void jacobi_eigen_mat3f(float *pMatrix, float *pdblVects, float *pdbEigenValues, float dbEps, int nJt)
{
    //-----------------------initiation--------------------------
    float A[9];
    float A1[9];
    for (int i = 0; i < 9; i++)
    {
        //initiate
        if (!isfinite(pMatrix[i]))
            return;
        A[i] = pMatrix[i];
        A1[i] = pMatrix[i];
    }

    float Vect_o[9] = { 0 };
    float Vect1[9] = { 0 };
    for (int i = 0; i < 3; i++) //initiate Eigen_Vector as unit matrix
    {
        Vect1[i * 3 + i] = 1.0f;
        Vect_o[i * 3 + i] = 1.0f;
    }

    int nCount = 0;		//iterations

    //-----------------------loop------------------------------
    while (1)
    {
        //find the maximum nondiagonal element A(np,nq) of matrix A
        float dbMax = A[1];
        int np = 0;
        int nq = 0;
        for (int i = 0; i < 3; i++)			//rows
        {
            for (int j = 0; j < i; j++)		//cols
            {
                float d = fabs(A[i * 3 + j]);
                if (d >= dbMax)
                {
                    dbMax = d;
                    np = i;
                    nq = j;
                }
            }
        }

        //Cyclic termination condition
        nCount++;

        if (nCount > nJt || dbMax < dbEps)
            break;

        //Jocobi Matrix Rotation
        float dbApp = A[np * 3 + np];
        float dbApq = A[np * 3 + nq];
        float dbAqq = A[nq * 3 + nq];

        //Angle calculation
        float dbAngle;
        if (dbAqq == dbApp)
        {
            dbAngle = ((dbApq) / fabs(dbApq)) * 3.1415926 / 4;
        }
        else
        {
            dbAngle = 0.5 * atan2(-2 * dbApq, (dbAqq - dbApp));
        }

        float dbSinTheta = sin(dbAngle);
        float dbCosTheta = cos(dbAngle);

        //Matrix update
        for (int i = 0; i < 3; i++)
        {
            if (i != np && i != nq)
            {
                A1[np * 3 + i] = A[np * 3 + i] * dbCosTheta + A[nq * 3 + i] * dbSinTheta; //A1pi = Api*cosX +Aqi*sinX
                A1[i * 3 + np] = A1[np * 3 + i]; //A1ip = A1pi

                A1[nq * 3 + i] = A[nq * 3 + i] * dbCosTheta - A[np * 3 + i] * dbSinTheta; //A1qi
                A1[i * 3 + nq] = A1[nq * 3 + i]; //A1iq
            }
        }
        A1[np * 3 + nq] = 0;
        A1[nq * 3 + np] = 0;

        A1[np * 3 + np] = dbApp * dbCosTheta * dbCosTheta + 2 * dbApq * dbSinTheta * dbCosTheta + dbAqq * dbSinTheta * dbSinTheta;
        A1[nq * 3 + nq] = dbAqq * dbCosTheta * dbCosTheta - 2 * dbApq * dbSinTheta * dbCosTheta + dbApp * dbSinTheta * dbSinTheta;

        //Eigen Vector
        for (int i = 0; i < 3; i++)
        {
            int u = i * 3 + np;		//ip
            int w = i * 3 + nq;		//iq
            float Vip = Vect_o[u];
            float Viq = Vect_o[w];
            Vect1[u] = Viq * dbSinTheta + Vip * dbCosTheta;
            Vect1[w] = Viq * dbCosTheta - Vip * dbSinTheta;
        }
        for (int i = 0; i < 9; i++)  	//update Vector
        {
            Vect_o[i] = Vect1[i];
        }
        for (int i = 0; i < 9; i++)
        {
            A[i] = A1[i];
        }
    }

    for (int i = 0; i < 3; i++)
    {
        pdbEigenValues[i] = A[i * 3 + i];
    }

    //----------------------sort-------------------------
    int sort_idx[3];
    for (int i = 0; i < 3; i++)
    {
        sort_idx[i] = i;
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = i + 1; j < 3; j++)
        {
            if (pdbEigenValues[i] < pdbEigenValues[j])
            {
                int temp, temp_idx;
                temp = pdbEigenValues[i];
                temp_idx = sort_idx[i];
                pdbEigenValues[i] = pdbEigenValues[j];
                sort_idx[i] = sort_idx[j];
                pdbEigenValues[j] = temp;
                sort_idx[j] = temp_idx;
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        int temp_col = sort_idx[i];
        for (int j = 0; j < 3; j++)
        {
            pdblVects[i + j * 3] = Vect_o[temp_col + j * 3];
        }
    }
}

__device__ void jacobi_eigen_mat4f(float *pMatrix, float *pdblVects, float *pdbEigenValues, float dbEps, int nJt)
{
    //-----------------------initiation--------------------------
    float A[16];
    float A1[16];
    for (int i = 0; i < 16; i++)
    {
        //initiate
        if (!isfinite(pMatrix[i]))
            return;
        A[i] = pMatrix[i];
        A1[i] = pMatrix[i];
    }

    float Vect_o[16] = { 0 };
    float Vect1[16] = { 0 };
    for (int i = 0; i < 4; i++) //initiate Eigen_Vector as unit matrix
    {
        Vect1[i * 4 + i] = 1.0f;
        Vect_o[i * 4 + i] = 1.0f;
    }

    int nCount = 0;		//iterations

    //-----------------------loop------------------------------
    while (1)
    {
        //find the maximum nondiagonal element A(np,nq) of matrix A
        float dbMax = A[1];
        int np = 0;
        int nq = 0;
        for (int i = 0; i < 4; i++)			//rows
        {
            for (int j = 0; j < i; j++)		//cols
            {
                float d = fabs(A[i * 4 + j]);
                if (d >= dbMax)
                {
                    dbMax = d;
                    np = i;
                    nq = j;
                }
            }
        }

        //Cyclic termination condition
        nCount++;

        if (nCount > nJt || dbMax < dbEps)
            break;

        //Jocobi Matrix Rotation
        float dbApp = A[np * 4 + np];
        float dbApq = A[np * 4 + nq];
        float dbAqq = A[nq * 4 + nq];

        //Angle calculation
        float dbAngle;
        if (dbAqq == dbApp)
        {
            dbAngle = ((dbApq) / fabs(dbApq)) * 3.1415926 / 4;
        }
        else
        {
            dbAngle = 0.5 * atan2(-2 * dbApq, (dbAqq - dbApp));
        }

        float dbSinTheta = sin(dbAngle);
        float dbCosTheta = cos(dbAngle);

        //Matrix update
        for (int i = 0; i < 4; i++)
        {
            if (i != np && i != nq)
            {
                A1[np * 4 + i] = A[np * 4 + i] * dbCosTheta + A[nq * 4 + i] * dbSinTheta; //A1pi = Api*cosX +Aqi*sinX
                A1[i * 4 + np] = A1[np * 4 + i]; //A1ip = A1pi

                A1[nq * 4 + i] = A[nq * 4 + i] * dbCosTheta - A[np * 4 + i] * dbSinTheta; //A1qi
                A1[i * 4 + nq] = A1[nq * 4 + i]; //A1iq
            }
        }
        A1[np * 4 + nq] = 0;
        A1[nq * 4 + np] = 0;

        A1[np * 4 + np] = dbApp * dbCosTheta * dbCosTheta + 2 * dbApq * dbSinTheta * dbCosTheta + dbAqq * dbSinTheta * dbSinTheta;
        A1[nq * 4 + nq] = dbAqq * dbCosTheta * dbCosTheta - 2 * dbApq * dbSinTheta * dbCosTheta + dbApp * dbSinTheta * dbSinTheta;

        //Eigen Vector
        for (int i = 0; i < 4; i++)
        {
            int u = i * 4 + np;		//ip
            int w = i * 4 + nq;		//iq
            float Vip = Vect_o[u];
            float Viq = Vect_o[w];
            Vect1[u] = Viq * dbSinTheta + Vip * dbCosTheta;
            Vect1[w] = Viq * dbCosTheta - Vip * dbSinTheta;
        }
        for (int i = 0; i < 16; i++)  	//update Vector
        {
            Vect_o[i] = Vect1[i];
        }
        for (int i = 0; i < 16; i++)
        {
            A[i] = A1[i];
        }
    }

    for (int i = 0; i < 4; i++)
    {
        pdbEigenValues[i] = A[i * 4 + i];
    }

    //----------------------sort-------------------------
    int sort_idx[4];
    for (int i = 0; i < 4; i++)
    {
        sort_idx[i] = i;
    }

    for (int i = 0; i < 4; i++)
    {
        for (int j = i + 1; j < 4; j++)
        {
            if (pdbEigenValues[i] < pdbEigenValues[j])
            {
                int temp, temp_idx;
                temp = pdbEigenValues[i];
                temp_idx = sort_idx[i];
                pdbEigenValues[i] = pdbEigenValues[j];
                sort_idx[i] = sort_idx[j];
                pdbEigenValues[j] = temp;
                sort_idx[j] = temp_idx;
            }
        }
    }

    for (int i = 0; i < 4; i++)
    {
        int temp_col = sort_idx[i];
        for (int j = 0; j < 4; j++)
        {
            pdblVects[i + j * 4] = Vect_o[temp_col + j * 4];
        }
    }
}

__global__ void eigCalcKernel(float *x, float *val, float *vec, int b, int d)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r < b)
    {
        if (d == 4)
            jacobi_eigen_mat4f(x + r * 16, vec + r * 16, val + r * 4, 1e-7, 20);
        else if (d == 3)
            jacobi_eigen_mat3f(x + r * 9, vec + r * 9, val + r * 3, 1e-7, 20);
    }
}

//x[B][3][3] val[B][3] vec[B][3][3]
//x[B][4][4] val[B][4] vec[B][4][4]
void calc_eig(float *x, float *val, float *vec, int b, int d)
{
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    int g = (b + 255) / 256;
    eigCalcKernel<<<g, 256, 0, s>>>(x, val, vec, b, d);
}
