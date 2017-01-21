#include "ThetaScheme.h"

#define CALL_OPTION 1 
#define A1 0.31938153
#define A2 -0.356563782
#define A3 1.781477937
#define A4 -1.821255978
#define A5 1.330274429
#define RSQRT2PI 0.39894228040143267793994605993438

float MoroInvCND(float P) {
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
    float y, z;

    if (P <= 0 || P >= 1.0f)
        return (float)(0x7FFFFFFF);

    y = P - 0.5f;
    if (fabsf(y) < 0.42f) {
        z = y * y;
        z = y * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
    }
    else {
        if (y > 0)
            z = logf(-logf(1.0f - P));
        else
            z = logf(-logf(P));

        z = c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)))))));
        if (y < 0) z = -z;
    }

    return z;
}

float Ih(float y1, float y2, float x1, float x2, float x)
{
    float y;
    y = (y2 * (x - x1) + y1 * (x2 - x)) / (x2 - x1);
    return y;
}

float function_f(float y, float z, float mu, float sigma, float r, float d)
{
    float f;
    f = (-r) * y - 1 / sigma * (mu - r + d) * z;
    return f;
}

void Make_grid(float *X, int M, float dh)
{
    int i;
    for (i = 0; i <= M; i++)
    {
        X[i] = (i - M / 2) * dh;
    }
}

/*---Terminal_condition--*/
void Terminal_condition(int M, float * X, float * YT, float S0, float T, float K,
    float sigma, float mu, float r, float d)
{
    float St;
    int i;
    for (i = 0; i <= M; i++)
    {
        St = S0 * expf(sigma * X[i] + (mu - 0.5 * sigma * sigma) * T);

        if (CALL_OPTION == 1)
        {
            YT[i] = max(St - K, 0.0);
        }
        else
        {
            YT[i] = max(K - St, 0.0);
        }
    }
}

void current_solution(int j, float *Y2, float *Z2, float *Y1, float *Z1, float *X, float th1, float th2, float dt, float dh, int NE, int N, float c, int M, float r, float sigma, float mu, float d, float *Random_matrix)
{
    int i, k, ii, a, Ps;

    float d_wt;
    float Sy, Sz, Syw, Sf, Sfw;
    float Xk;
    float Ey, Ez, Eyw, Ef, Efw;
    int size = (M + 1);
    unsigned int seed = 1;
    float sq = sqrtf(dt);
    Ps = M / (2 * N);
    ii = Ps * (N - j);

    for (i = ii; i <= M - ii; i++)
    {
        Ey = Ez = Eyw = Ef = Efw = 0;
        for (k = 1; k <= NE; k++)
        {
            d_wt = Random_matrix[k];

            Xk = X[i] + d_wt;

            if (Xk < X[i - Ps])
                Xk = X[i - Ps];
            else if (Xk > X[i + Ps])
                Xk = X[i + Ps];

            a = (Xk - X[0]) / dh;
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

            Syw = Sy * d_wt;
            Sf = function_f(Sy, Sz, mu, sigma, r, d);
            Sfw = Sf * d_wt;

            Ey = Sy + Ey;
            Ez = Sz + Ez;
            Eyw = Syw + Eyw;
            Ef = Sf + Ef;
            Efw = Sfw + Efw;
        }
        Z2[i] = (Eyw + dt * (1 - th2) * Efw - dt * (1 - th2) * Ez) / (NE * dt * th2);
        Y2[i] = ((Ey + dt * (1 - th1) * Ef) / NE - dt * th1 * (1 / sigma) * (mu - r + d) * Z2[i]) / (1 + dt * th1 * r);
    }
}

void print_solution(float y, float z)
{
    printf("\n");
    printf("The value of Y0 is: %10.4f\n", y);
}
