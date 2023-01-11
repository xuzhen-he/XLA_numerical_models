#include <chrono>
#include <string>
#include <omp.h>
#include <thread>
#include "vec_bench.h"

#define value_t float
#define index_t int

#define XPXPY 20

struct vec_xpxpy : public vec_bench<value_t, index_t>
{

    void benchmark()
    {
        print_bench();

#if XPXPY == 6
        std::cout << "\nSimulation info: 1d vec xpxpy 6\n";
#elif XPXPY == 12
        std::cout << "\nSimulation info: 1d vec xpxpy 12\n";
#elif XPXPY == 20
        std::cout << "\nSimulation info: 1d vec xpxpy 20\n";
#else
#endif

        value_t *x = new value_t[total_size];
        value_t *y = new value_t[total_size];

#pragma omp parallel for
        for (index_t j = 0; j < total_size; j++)
        {
            x[j] = 1.0;
            y[j] = 0.0;
        }

        loops = 0;
        auto start = std::chrono::high_resolution_clock::now();
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count()) < 1000.0 * benchtime)
        {
// #pragma omp simd
#pragma omp parallel for
            for (index_t j = 0; j < total_size; j++)
            {
#if XPXPY == 6
                y[j] += x[j] - x[j] + x[j] - x[j] + x[j] - x[j];
#elif XPXPY == 12
                y[j] += x[j] - x[j] + x[j] - x[j] + x[j] - x[j] 
                        + x[j] - x[j] + x[j] - x[j] + x[j] - x[j];
#elif XPXPY == 20
                y[j] += x[j] - x[j] + x[j] - x[j] + x[j] - x[j] 
                        + x[j] - x[j] + x[j] - x[j] + x[j] - x[j]
                        + x[j] - x[j] + x[j] - x[j] + x[j] - x[j]
                        + x[j] - x[j];
#else

#endif
            }

            loops++;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        duration = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        test_result(y, 0.0);
        print_performance();

        delete[] x;
        delete[] y;
    }

    vec_xpxpy(int narg, char **arg) : vec_bench<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 3.0 * sizeof(value_t) * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);
    }
};

int main(int narg, char **arg)
{
    vec_xpxpy test(narg, arg);
    test.benchmark();
}