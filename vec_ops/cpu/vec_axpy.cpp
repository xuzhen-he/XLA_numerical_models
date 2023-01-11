#include <chrono>
#include <string>
#include <omp.h>
#include <thread>
#include "vec_bench.h"

#define value_t float
#define index_t int

struct vec_axpy : public vec_bench<value_t, index_t>
{

    void benchmark()
    {
        print_bench();

        std::cout << "\nSimulation info: 1d vec axpy\n";

        value_t *x = new value_t[total_size];
        value_t *y = new value_t[total_size];
        value_t a = 1.0;

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
                y[j] += a * x[j];
            loops++;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        duration = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        value_t target = (value_t)loops * (value_t)total_size;
        test_result(y, target);
        print_performance();

        delete[] x;
        delete[] y;
    }

    vec_axpy(int narg, char **arg) : vec_bench<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 3.0 * sizeof(value_t) * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);
    }
};

int main(int narg, char **arg)
{
    vec_axpy test(narg, arg);
    test.benchmark();
}