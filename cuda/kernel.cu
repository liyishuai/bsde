
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../cub/cub/cub.cuh"

#include <Windows.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CALL_OPTION 1

#define EXPIRATION_TIME .33f
#define STRIKE_PRICE    100.f
#define INIT_PRICE      100.f

#define sigma   .2f
#define r       .03f
#define R       .03f
#define mu      .05f
#define d       .04f

#define THREADS_PER_BLOCK 1024

__global__ void makeGrid(float *X, int M, float dh)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i <= M)
        X[i] = (i - M / 2) * dh;
}

__global__ void terminalCondition(int M, const float *X, float *YT, float S0, float T, float K)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i <= M)
    {
        const float St = S0 * expf(sigma * X[i] + (mu - .5f * sigma * sigma) * T);
        YT[i] = fmaxf(CALL_OPTION == 1 ? St - K : K - St, 0.f);
    }
}

__global__ void moroInvCND(float *randomMatrix, int num, float sqrtdt)
{
    const float a1 = 2.50662823884f;
    const float a2 = -18.61500062529f;
    const float a3 = 41.39119773534f;
    const float a4 = -25.44106049637f;
    const float b1 = -8.4735109309f;
    const float b2 = 23.08336743743f;
    const float b3 = -21.06224101826f;
    const float b4 = 3.13082909833f;
    const float c1 = 0.337475482272615f;
    const float c2 = 0.976169019091719f;
    const float c3 = 0.160797971491821f;
    const float c4 = 2.76438810333863E-02f;
    const float c5 = 3.8405729373609E-03f;
    const float c6 = 3.951896511919E-04f;
    const float c7 = 3.21767881768E-05f;
    const float c8 = 2.888167364E-07f;
    const float c9 = 3.960315187E-07f;

    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i + 1 < num)
    {
        const float P = (i + 1) / (float)num;
        const float y = P - .5f;
        float z;
        if (fabsf(y) < .42f)
        {
            z = y * y;
            z = y * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.f);
        }
        else
        {
            z = y > 0 ? logf(-logf(1.f - P)) : logf(-logf(P));
            z = c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)))))));
            if (y < 0)
                z = -z;
        }
        randomMatrix[i] = z * sqrtdt;
    }
}

struct E
{
    float y;
    float z;
    float yw;
    float f;
    float fw;
    
    __host__ __device__ E(float yy = 0.f,
        float zz = 0.f,
        float yww = 0.f,
        float ff = 0.f,
        float fww = 0.f) :
        y(yy),
        z(zz),
        yw(yww),
        f(ff),
        fw(fww) {}

    __host__ __device__ E operator +(const E &e) const
    {
        return E{ y + e.y, z + e.z, yw + e.yw, f + e.f, fw + e.fw };
    }
};

#define Ih(y1,y2,x1,x2,x) (y2 * (x - x1) + y1 * (x2 - x)) / (x2 - x1)

#define function_f(y,z) (-r) * y - 1 / sigma * (mu - r + d) * z

__global__ void calculate(int i, E *e, const float *X, const float *Y1, const float *Z1, const float *randomMatrix, int NE, int Ps, float dh)
{
    const int k = threadIdx.x + blockDim.x * blockIdx.x;
    if (k < NE)
    {
        const float d_wt = randomMatrix[k];
        float Xk = X[i] + d_wt;

        if (Xk < X[i - Ps])
            Xk = X[i - Ps];
        else if (Xk > X[i + Ps])
            Xk = X[i + Ps];

        const int a = (Xk - X[0]) / dh;
        float Sy;
        float Sz;
        if (a == i + Ps)
        {
            Sy = Y1[a];
            Sz = Z1[a];
        }
        else
        {
            Sy = Ih(Y1[a], Y1[a + 1], X[a], X[a + 1], Xk);
            Sz = Ih(Z1[a], Z1[a + 1], X[a], X[a + 1], Xk);
        }
        float Sf = function_f(Sy, Sz);
        e[k].y = Sy;
        e[k].z = Sz;
        e[k].yw = Sy * d_wt;
        e[k].f = Sf;
        e[k].fw = Sf * d_wt;
    }
}

float dt;
float dh;
float c;
float *X;
float *randomMatrix;
E *e, *er, erh;
void  *d_temp_storage = NULL;
size_t temp_storage_bytes = 0;

int NE;
int N;
int M;
int Ps;

void currentSolution(int j, float *Y2, float *Z2, float *Y1, float *Z1, float th1, float th2)
{
    const int ii = Ps * (N - j);
    for (int i = ii; i <= M - ii; ++i)
    {
        calculate<<<(NE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(i, e,  X, Y1, Z1, randomMatrix, NE, Ps, dh);
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "calculate failed!");
            goto Error;
        }
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, e, er, NE);
        cudaMemcpy(&erh, er, sizeof(E), cudaMemcpyDeviceToHost);
        const float newZ2i = (erh.yw + dt * (1.f - th2) * erh.fw - dt * (1.f - th2) * erh.z) / (NE * dt * th2);
        const float newY2i = ((erh.y + dt * (1.f - th1) * erh.f) / NE - dt * th1 * (1.f / sigma) * (mu - r + d) * newZ2i) / (1.f + dt * th1 * r);
        cudaMemcpy(Z2 + i, &newZ2i, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Y2 + i, &newY2i, sizeof(float), cudaMemcpyHostToDevice);
    }
Error:
}

int main(int argc, char *argv[])
{
    int TIME_GRID = 64;
    if (argc >= 2)
        TIME_GRID = atoi(argv[1]);

    int SIM_TIMES = 40000;
    if (argc >= 3)
        SIM_TIMES = atoi(argv[2]);

    printf("TIME_GRID = %d\n", TIME_GRID);
    float S = INIT_PRICE;
    float K = STRIKE_PRICE;
    float T = EXPIRATION_TIME;

    N = TIME_GRID;
    dt = T / N;
    dh = dt;
    c = 5 * sqrtf(dt);
    printf("c=%f\n", c);
    
    Ps = c / dh + 1;
    M = N * Ps * 2;
    NE = SIM_TIMES;

    int size = (M + 1) * sizeof(float);

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&X, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc X failed!");
        goto Error;
    }
    makeGrid<<<M / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(X, M, dh);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "makeGrid failed!");
        goto Error;
    }
    float *Y1;
    cudaStatus = cudaMalloc((void**)&Y1, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc Y1 failed!");
        goto Error;
    }
    terminalCondition<<<M / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(M, X, Y1, S, T, K);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "terminalCondition failed!");
        goto Error;
    }
    float *Y2;
    cudaStatus = cudaMalloc((void**)&Y2, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc Y2 failed!");
        goto Error;
    }
    float *Z1;
    cudaStatus = cudaMalloc((void**)&Z1, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc Z1 failed!");
        goto Error;
    }
    float *Z2;
    cudaStatus = cudaMalloc((void**)&Z2, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc Z2 failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&randomMatrix, NE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc randomMatrix failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&e, NE * sizeof(E));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc e failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&er, sizeof(E));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc er failed!");
        goto Error;
    }
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, e, er, NE);
    cudaStatus = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_temp_storage failed!");
        goto Error;
    }

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);

    LARGE_INTEGER start;
    LARGE_INTEGER finish;
    float solution;
    QueryPerformanceCounter(&start);

    int num = NE + 1;
    moroInvCND<<<(NE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(randomMatrix, num, sqrtf(dt));
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "moroInvCND failed!");
        goto Error;
    }

    int j;
    for (j = N - 1; j >= 0; j -= 2)
    {
        float th1 = 0.5;
        float th2 = 0.5;
        if (j == N - 1)
            th1 = th2 = 1;

        currentSolution(j, Y2, Z2, Y1, Z1, th1, th2);

        th1 = th2 = 0.5;

        if (j > 0)
            currentSolution(j - 1, Y1, Z1, Y2, Z2, th1, th2);

        printf("step.%d finish\n", j);
    }
    if (j == -1)
        cudaMemcpy(&solution, Y1 + M / 2, sizeof(float), cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(&solution, Y2 + M / 2, sizeof(float), cudaMemcpyDeviceToHost);

    QueryPerformanceCounter(&finish);
    float tm = finish.QuadPart - start.QuadPart;
    tm /= frequency.QuadPart;
    printf("FINISHED!!\nALL Time is %.6f s\n", tm);
    printf("\nThe value of Y0 is: %10.4f\n", solution);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;

Error:
    cudaFree(X);
    cudaFree(Y1);
    cudaFree(Y2);
    cudaFree(Z1);
    cudaFree(Z2);
    cudaFree(randomMatrix);
    cudaFree(e);
    cudaFree(er);
    cudaFree(d_temp_storage);
    return 1;
}