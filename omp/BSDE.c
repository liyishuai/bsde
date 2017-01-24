#include "ThetaScheme.h"

int SIM_TIMES = 40000;
#define EXPIRATION_TIME 0.33
#define STRIKE_PRICE    100
#define INIT_PRICE      100

int main(int argc, char* argv[])
{
    float S, K, T, sigma, r, R, mu, d;
    int TIME_GRID = 64;

    if (argc >= 2)
        TIME_GRID = atoi(argv[1]);
    if (argc >= 3)
        SIM_TIMES = atoi(argv[2]);

    printf("TIME_GRID = %d\n", TIME_GRID);
    S = INIT_PRICE;
    K = STRIKE_PRICE;
    T = EXPIRATION_TIME;

    sigma = 0.2;
    r = 0.03;
    R = 0.03;
    mu = 0.05;
    d = 0.04;

    int N, M;
    int NE;
    float dt, dh;
    float th1 = 0.0, th2 = 0.0;
    float c;
    int Ps;

    N = TIME_GRID;
    dt = T / N;

    dh = dt;
    c = 5.0 * sqrtf(dt);
    printf("c=%f\n", c);
    Ps = c / dh + 1;
    M = N * Ps * 2;
    NE = SIM_TIMES;

    float * X;
    int size = (M + 1) * sizeof(float);
    X = (float*)malloc(size);

    float * Y1, *Y2, *Z1, *Z2;
    Y1 = (float*)malloc(size);
    Y2 = (float*)malloc(size);
    Z1 = (float*)malloc(size);
    Z2 = (float*)malloc(size);

    memset(Y2, 0, size);
    memset(Z2, 0, size);
    memset(Z1, 0, size);

    Make_grid(X, M, dh);

    Terminal_condition(M, X, Y1, S, T, K, sigma, mu, r, d);

    int j = 0;

    LARGE_INTEGER start;
    LARGE_INTEGER finish;
    LARGE_INTEGER frequency;
    float tm;

    int k;
    unsigned int seed = (unsigned int)time(NULL);
    float *Random_matrix;
    int num = NE + 1;
    Random_matrix = (float *)malloc(sizeof(float)*(num));
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    for (k = 1; k <= num; k++)
    {
        Random_matrix[k] = MoroInvCND((float)k / (NE + 1))*sqrt(dt);
    }

    for (j = N - 1; j >= 0; j -= 2)
    {

        if (j == N - 1)
            th1 = th2 = 1;
        else
            th1 = th2 = 0.5;

        current_solution(j, Y2, Z2, Y1, Z1, X, th1, th2, dt, dh, NE, N, c, M, r, sigma, mu, d, Random_matrix);

        th1 = th2 = 0.5;

        if (j > 0)
        {
            current_solution(j - 1, Y1, Z1, Y2, Z2, X, th1, th2, dt, dh, NE, N, c, M, r, sigma, mu, d, Random_matrix);
        }
        else
        {
            break;
        }
        printf("step.%d finish\n", j);
    }

    QueryPerformanceCounter(&finish);
    tm = finish.QuadPart - start.QuadPart;
    tm /= frequency.QuadPart;
    printf("FINISHED!!\nALL Time is %.6f s\n", tm);

    if (j == -1)
        print_solution(Y1[M / 2], Z1[M / 2]);
    else
        print_solution(Y2[M / 2], Z2[M / 2]);

    free(X);
    free(Random_matrix);
    free(Y1);
    free(Z1);
    free(Y2);
    free(Z2);
    return 0;
}
