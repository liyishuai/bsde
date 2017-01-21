#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <Windows.h>

float MoroInvCND(float P);

float Ih(float y1, float y2, float x1, float x2, float x);

float function_f(float y, float z, float mu, float sigma, float r, float d);

void Make_grid(float *X, int M, float dh);

void Terminal_condition(int M, float * X, float * YT, float S0, float T, float K, float sigma, float mu, float r, float d);

void current_solution(int j, float *Y2, float *Z2, float*Y1, float*Z1, float *X, float th1, float th2, float dt, float dh, int NE, int N, float c, int M, float r, float sigma, float mu, float d, float *Random_matrix);

void print_solution(float y, float z);
