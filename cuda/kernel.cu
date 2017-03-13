
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

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

    __device__ E() : y(0), z(0), yw(0), f(0), fw(0) {}

    __device__ E& operator +=(const E &e)
    {
        y += e.y;
        z += e.z;
        yw += e.yw;
        f += e.f;
        fw += e.fw;
        return *this;
    }
};

#define Ih(y1,y2,x1,x2,x) (y2 * (x - x1) + y1 * (x2 - x)) / (x2 - x1)

#define function_f(y,z) (-r) * y - 1 / sigma * (mu - r + d) * z

__global__ void calculate(int ii, const float *X, float *Y2, float *Z2, const float *Y1, const float *Z1, float th1, float th2, const float *randomMatrix, int NE, int Ps, float dt, float dh)
{
    const int i = ii + blockIdx.x;
    const int x = threadIdx.x;
    __shared__ E e[THREADS_PER_BLOCK];
    for (int k = x; k < NE; k += blockDim.x)
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
        e[x].y += Sy;
        e[x].z += Sz;
        e[x].yw += Sy * d_wt;
        e[x].f += Sf;
        e[x].fw += Sf * d_wt;
    }
    __syncthreads();
    if (x < 512)
    {
        e[x] += e[x + 512];
        __syncthreads();
    }
    if (x < 256)
    {
        e[x] += e[x + 256];
        __syncthreads();
    }
    if (x < 128)
    {
        e[x] += e[x + 128];
        __syncthreads();
    }
    if (x < 64)
    {
        e[x] += e[x + 64];
        __syncthreads();
    }
    if (x < 32)
    {
        e[x] += e[x + 32];
        e[x] += e[x + 16];
        e[x] += e[x + 8];
        e[x] += e[x + 4];
        e[x] += e[x + 2];
        e[x] += e[x + 1];
    }
    if (x == 0)
    {
        Z2[i] = (e[0].yw + dt * (1.f - th2) * e[0].fw - dt * (1.f - th2) * e[0].z) / (NE * dt * th2);
        Y2[i] = ((e[0].y + dt * (1.f - th1) * e[0].f) / NE - dt * th1 * (1.f / sigma) * (mu - r + d) * Z2[i]) / (1.f + dt * th1 * r);
    }
}

float dt;
float dh;
float c;
float *X;
float *randomMatrix;

int NE;
int N;
int M;
int Ps;

void currentSolution(int j, float *Y2, float *Z2, float *Y1, float *Z1, float th1, float th2)
{
    const int ii = Ps * (N - j);
    calculate << <M - ii - ii + 1, THREADS_PER_BLOCK >> > (ii, X, Y2, Z2, Y1, Z1, th1, th2, randomMatrix, NE, Ps, dt, dh);
    checkCudaErrors(cudaGetLastError());
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

    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMalloc((void**)&X, size));

    makeGrid << <M / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (X, M, dh);
    checkCudaErrors(cudaGetLastError());

    float *Y1;
    checkCudaErrors(cudaMalloc((void**)&Y1, size));

    terminalCondition << <M / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (M, X, Y1, S, T, K);
    checkCudaErrors(cudaGetLastError());

    float *Y2;
    checkCudaErrors(cudaMalloc((void**)&Y2, size));

    float *Z1;
    checkCudaErrors(cudaMalloc((void**)&Z1, size));

    float *Z2;
    checkCudaErrors(cudaMalloc((void**)&Z2, size));

    checkCudaErrors(cudaMalloc((void**)&randomMatrix, NE * sizeof(float)));

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);

    LARGE_INTEGER start;
    LARGE_INTEGER finish;
    float solution;
    QueryPerformanceCounter(&start);

    int num = NE + 1;
    moroInvCND << <(NE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (randomMatrix, num, sqrtf(dt));
    checkCudaErrors(cudaGetLastError());

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
    checkCudaErrors(cudaDeviceReset());
    
    checkCudaErrors(cudaFree(X));
    checkCudaErrors(cudaFree(Y1));
    checkCudaErrors(cudaFree(Y2));
    checkCudaErrors(cudaFree(Z1));
    checkCudaErrors(cudaFree(Z2));
    checkCudaErrors(cudaFree(randomMatrix));
    return 0;
}