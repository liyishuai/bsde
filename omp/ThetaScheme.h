#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>
#include <ctime>
#include <chrono>
#include <string>
using namespace std;

float MoroInvCND(const float &P);

float Ih(const float &y1, const float &y2, const float &x1, const float &x2, const float &x);

float function_f(const float &y, const float &z, const float &mu, const float &sigma, const float &r, const float &R, const float &d);

void Make_grid(float *X, const int &M, const float &dh);

void Terminal_condition(const int &M, float * X, float * YT, const float &S0, const float &T, const float &K, const bool &call_option, const float &sigma, const float &mu, const float &r, const float &d);

void current_solution(const int &j, float *Y2, float *Z2, const float*Y1, const float*Z1, const float *X, const float &th1, const float &th2, const float &dt, const float &dh, const int &NE, const int &N, const float &c, const int &M, const float &r, const float &R, const float &sigma, const float &mu, const float &d, const float *Random_matrix);

void print_solution(const float &tm, const string &name, const float &y, const float &z);
