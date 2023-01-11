#include <chrono>
#include <cmath>
#include <string>
#include <omp.h>
#include <thread>
#include "diff1d.h"

#define value_t float
#define index_t int

struct diff1d_cpu : public diff1d<value_t, index_t>
{
    void benchmark()
    {
        print_bench();

        value_t *u = new value_t[total_size];
        value_t *u_new = new value_t[total_size];

        initial_condition(u, u_new);

        loops = 0;
        auto start = std::chrono::high_resolution_clock::now();
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count()) < 1000.0 * benchtime)
        {
// #pragma omp parallel for
#pragma omp simd
            for (index_t j = 1; j < total_size - 1; j++)
                u_new[j] = (1.0 - 2.0 * r) * u[j] + r * (u[j - 1] + u[j + 1]);

            u_new[0] = 0.0;
            u_new[total_size - 1] = 0.0;

            // swap u and u_new
            value_t *tmp = u;
            u = u_new;
            u_new = tmp;

            loops++;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        duration = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        value_t t = delta_t * value_t(loops);
        test_result(u, t);
        print_performance();

        delete[] u;
        delete[] u_new;
    }

    diff1d_cpu(int narg, char **arg) : diff1d<value_t, index_t>(narg, arg) {}
};

int main(int narg, char **arg)
{
    diff1d_cpu test(narg, arg);
    test.benchmark();
}