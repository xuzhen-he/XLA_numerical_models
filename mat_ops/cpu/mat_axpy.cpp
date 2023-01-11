#include <chrono>
#include <string>
#include <omp.h>
#include <thread>
#include "array2d.h"
#include "mat_bench.h"

#define value_t float
#define index_t int

struct mat_axpy : public mat_bench<value_t, index_t>
{

    void benchmark()
    {
        print_bench();

        std::cout << "\nSimulation info: 2d mat axpy\n";

        value_t **x = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **y = create_array2d<value_t, index_t>(side_size, side_size);
        value_t a = 1.0;

#pragma omp parallel for
        for (index_t i = 0; i < side_size; i++)
        {
            for (index_t j = 0; j < side_size; j++)
            {
                x[i][j] = 1.0;
                y[i][j] = 0.0;
            }
        }

        loops = 0;
        auto start = std::chrono::high_resolution_clock::now();
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count()) < 1000.0 * benchtime)
        {

#pragma omp parallel for
            for (index_t i = 0; i < side_size; i++)
            {
                // #pragma omp simd
                for (index_t j = 0; j < side_size; j++)
                {
                    y[i][j] += a * x[i][j];
                }
            }
            loops++;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        duration = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        value_t target = value_t(total_size * loops);
        test_result(y, target);
        print_performance();

        destroy_array2d<value_t, index_t>(x);
        destroy_array2d<value_t, index_t>(y);
    }

    mat_axpy(int narg, char **arg) : mat_bench<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 3.0 * sizeof(value_t) * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);
    }
};

int main(int narg, char **arg)
{
    mat_axpy test(narg, arg);
    test.benchmark();
}