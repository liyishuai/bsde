
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_timer.h"
#include "multithreading.h"

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
using namespace std;

#define THREADS_PER_BLOCK 128

__global__ void makeGrid(float *X, int M, float dh)
{
    const int i(threadIdx.x + blockDim.x * blockIdx.x);
    if (i <= M)
        X[i] = (i - M / 2) * dh;
}

__global__ void terminalCondition(int M, const float *X, float *YT, float S0, float T, float K, bool call_option, float sigma, float mu)
{
    const int i(threadIdx.x + blockDim.x * blockIdx.x);
    if (i <= M)
    {
        const float St(S0 * expf(sigma * X[i] + (mu - .5f * sigma * sigma) * T));
        YT[i] = fmaxf(call_option ? St - K : K - St, 0.f);
    }
}

#define a1 2.50662823884f
#define a2 -18.61500062529f
#define a3 41.39119773534f
#define a4 -25.44106049637f
#define b1 -8.4735109309f
#define b2 23.08336743743f
#define b3 -21.06224101826f
#define b4 3.13082909833f
#define c1 0.337475482272615f
#define c2 0.976169019091719f
#define c3 0.160797971491821f
#define c4 2.76438810333863E-02f
#define c5 3.8405729373609E-03f
#define c6 3.951896511919E-04f
#define c7 3.21767881768E-05f
#define c8 2.888167364E-07f
#define c9 3.960315187E-07f

__global__ void moroInvCND(float *randomMatrix, int num, float sqrtdt)
{
    const int i(threadIdx.x + blockDim.x * blockIdx.x);
    if (i + 1 < num)
    {
        const float P((i + 1) / (float)num);
        const float y(P - .5f);
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

#define function_f(y,z,mu,sigma,r,R,d) -(r * y + (mu - r + d) * z / sigma + (R - r) * fminf(y - z, 0.f))

__global__ void calculate(int ii, const float *X, float *Y2, float *Z2, const float *Y1, const float *Z1, float th1, float th2, const float *randomMatrix, int NE, int Ps, float dt, float dh, float r, float R, float sigma, float mu, float d)
{
    const int i(ii + blockIdx.x);
    const int x(threadIdx.x);
    __shared__ E e[THREADS_PER_BLOCK];
    for (int k(x); k < NE; k += blockDim.x)
    {
        const float d_wt(randomMatrix[k]);
        float Xk(X[i] + d_wt);

        if (Xk < X[i - Ps])
            Xk = X[i - Ps];
        else if (Xk > X[i + Ps])
            Xk = X[i + Ps];

        const int a((Xk - X[0]) / dh);
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

        float Sf(function_f(Sy, Sz, mu, sigma, r, R, d));
        e[x].y += Sy;
        e[x].z += Sz;
        e[x].yw += Sy * d_wt;
        e[x].f += Sf;
        e[x].fw += Sf * d_wt;
    }
    __syncthreads();
    if (THREADS_PER_BLOCK >= 1024 && x < 512)
    {
        e[x] += e[x + 512];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 512 && x < 256)
    {
        e[x] += e[x + 256];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 256 && x < 128)
    {
        e[x] += e[x + 128];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 128 && x < 64)
    {
        e[x] += e[x + 64];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 64 && x < 32)
    {
        e[x] += e[x + 32];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 32 && x < 16)
    {
        e[x] += e[x + 16];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 16 && x < 8)
    {
        e[x] += e[x + 8];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 8 && x < 4)
    {
        e[x] += e[x + 4];
        __syncthreads();
    }
    if (THREADS_PER_BLOCK >= 4 && x < 2)
    {
        e[x] += e[x + 2];
        __syncthreads();
    }
    if (x == 0)
    {
        if (THREADS_PER_BLOCK >= 2)
            e[0] += e[1];
        Z2[i] = (e[0].yw + dt * (1.f - th2) * e[0].fw - dt * (1.f - th2) * e[0].z) / (NE * dt * th2);
        Y2[i] = ((e[0].y + dt * (1.f - th1) * e[0].f) / NE - dt * th1 * (1.f / sigma) * (mu - r + d) * Z2[i]) / (1.f + dt * th1 * r);
    }
}

void currentSolution(const int &j, float *Y2, float *Z2, const float *Y1, const float *Z1, const float *X, const float &th1, const float &th2, const float &dt, const float &dh, const int &NE, const int &N, const int &M, const int &Ps, const float &r, const float &R, const float &sigma, const float &mu, const float &d, const float *randomMatrix)
{
    const int ii(Ps * (N - j));
    calculate << <M - ii - ii + 1, THREADS_PER_BLOCK >> > (ii, X, Y2, Z2, Y1, Z1, th1, th2, randomMatrix, NE, Ps, dt, dh, r, R, sigma, mu, d);
    checkCudaErrors(cudaGetLastError());
}

struct config
{
    string name;
    bool call_option;
    float S;
    float K;
    float T;
    float sigma;
    float r;
    float R;
    float mu;
    float d;
};

istream& operator >> (istream &in, config &c)
{
    return in >> c.name >> c.call_option >> c.S >> c.K >> c.T >> c.sigma >> c.r >> c.R >> c.mu >> c.d;
}

struct configs : vector<config>
{
    int device;

    void set_device(const int &dev)
    {
        device = dev;
    }
};

int TIME_GRID;
int SIM_TIMES;
int N, NE;
float *randomMatrix;

CUT_THREADPROC solverThread(configs *cfgs)
{
    checkCudaErrors(cudaSetDevice(cfgs->device));
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    for (const config &cfg : *cfgs)
    {
        const float dt(cfg.T / N);
        const float dh(dt);
        const float c(5 * sqrtf(dt));
        const int Ps(c / dh + 1);
        const int M(N * Ps * 2);
        const int size((M + 1) * sizeof(float));

        float *X;
        checkCudaErrors(cudaSetDevice(0));
        checkCudaErrors(cudaMalloc((void**)&X, size));

        makeGrid << <M / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (X, M, dh);
        checkCudaErrors(cudaGetLastError());

        float *Y1;
        checkCudaErrors(cudaMalloc((void**)&Y1, size));

        terminalCondition << <M / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (M, X, Y1, cfg.S, cfg.T, cfg.K, cfg.call_option, cfg.sigma, cfg.mu);
        checkCudaErrors(cudaGetLastError());

        float *Y2;
        checkCudaErrors(cudaMalloc((void**)&Y2, size));

        float *Z1;
        checkCudaErrors(cudaMalloc((void**)&Z1, size));

        float *Z2;
        checkCudaErrors(cudaMalloc((void**)&Z2, size));

        float solutionY, solutionZ;
        auto start(chrono::high_resolution_clock::now());

        moroInvCND << <(NE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (randomMatrix, NE + 1, sqrtf(dt));
        checkCudaErrors(cudaGetLastError());

        int j;
        for (j = N - 1; j >= 0; j -= 2)
        {
            float th1 = 0.5;
            float th2 = 0.5;
            if (j == N - 1)
                th1 = th2 = 1;

            currentSolution(j, Y2, Z2, Y1, Z1, X, th1, th2, dt, dh, NE, N, M, Ps, cfg.r, cfg.R, cfg.sigma, cfg.mu, cfg.d, randomMatrix);

            th1 = th2 = 0.5;

            if (j > 0)
                currentSolution(j - 1, Y1, Z1, Y2, Z2, X, th1, th2, dt, dh, NE, N, M, Ps, cfg.r, cfg.R, cfg.sigma, cfg.mu, cfg.d, randomMatrix);
        }
        if (j == -1)
        {
            cudaMemcpy(&solutionY, Y1 + M / 2, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&solutionZ, Z1 + M / 2, sizeof(float), cudaMemcpyDeviceToHost);
        }
        else
        {
            cudaMemcpy(&solutionY, Y2 + M / 2, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&solutionZ, Z2 + M / 2, sizeof(float), cudaMemcpyDeviceToHost);
        }

        float tm(sdkGetTimerValue(&timer) * 1e-3);
        cout << tm << '\t' << cfg.name << '\t' << solutionY << '\t' << solutionZ << endl;

        checkCudaErrors(cudaFree(X));
        checkCudaErrors(cudaFree(Y1));
        checkCudaErrors(cudaFree(Z1));
        checkCudaErrors(cudaFree(Y2));
        checkCudaErrors(cudaFree(Z2));
    }
    sdkDeleteTimer(&timer);
    checkCudaErrors(cudaDeviceReset());
}

int main(int argc, char *argv[])
{
    cin >> SIM_TIMES >> TIME_GRID;
    N = TIME_GRID;
    NE = SIM_TIMES;

    checkCudaErrors(cudaMalloc((void**)&randomMatrix, SIM_TIMES * sizeof(float)));

    int GPU_N;
    checkCudaErrors(cudaGetDeviceCount(&GPU_N));

    configs *cfgs = new configs[GPU_N];
    config cfg;
    for (int i = 0; cin >> cfg; i %= GPU_N)
        cfgs[i++].push_back(cfg);

    CUTThread *threadID = new CUTThread[GPU_N];
    for (int i = 0; i < GPU_N; ++i)
    {
        cfgs[i].set_device(i);
        threadID[i] = cutStartThread((CUT_THREADROUTINE)solverThread, cfgs + i);
    }
    cutWaitForThreads(threadID, GPU_N);

    delete[] threadID;
    delete[] cfgs;

    return 0;
}
