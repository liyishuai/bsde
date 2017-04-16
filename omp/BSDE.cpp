#include "ThetaScheme.h"

#define EXPIRATION_TIME 0.33
#define STRIKE_PRICE    100
#define INIT_PRICE      100

int main(int argc, char* argv[])
{
    float S, K, T, sigma, r, R, mu, d;
    int SIM_TIMES;
    int TIME_GRID;
    cin >> SIM_TIMES >> TIME_GRID;
    const int N = TIME_GRID;

    while (cin >> S >> K >> T >> sigma >> r >> R >> mu >>d)
    {
        float th1 = 0.0, th2 = 0.0;

        const float dt = T / N;
        const float dh = dt;

        const float c = 5.0 * sqrtf(dt);
        const float Ps = c / dh + 1;
        const int M = N * Ps * 2;
        const int NE = SIM_TIMES;

        int size = M + 1;
        float *X  = new float[size];
        float *Y1 = new float[size];
        float *Y2 = new float[size];
        float *Z1 = new float[size];
        float *Z2 = new float[size];

        Make_grid(X, M, dh);

        Terminal_condition(M, X, Y1, S, T, K, sigma, mu, r, d);

        float *Random_matrix;
        int num = NE + 1;
        Random_matrix = (float *)malloc(sizeof(float)*(num));
        auto start(chrono::high_resolution_clock::now());
        for (int k = 1; k <= num; k++)
            Random_matrix[k] = MoroInvCND((float)k / (NE + 1))*sqrt(dt);

        int j;

        for (j = N - 1; j >= 0; j -= 2)
        {

            if (j == N - 1)
                th1 = th2 = 1;
            else
                th1 = th2 = 0.5;

            current_solution(j, Y2, Z2, Y1, Z1, X, th1, th2, dt, dh, NE, N, c, M, r, sigma, mu, d, Random_matrix);

            th1 = th2 = 0.5;

            if (j > 0)
                current_solution(j - 1, Y1, Z1, Y2, Z2, X, th1, th2, dt, dh, NE, N, c, M, r, sigma, mu, d, Random_matrix);
            else
                break;
        }

        chrono::duration<double> tm(chrono::high_resolution_clock::now() - start);
        cerr << tm.count() << endl;

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
    }
    return 0;
}
