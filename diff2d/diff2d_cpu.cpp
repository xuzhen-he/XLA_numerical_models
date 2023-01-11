#include <chrono>
#include <cmath>
#include <string>
#include <omp.h>
#include <thread>
#include "diff2d.h"
#include "array2d.h"

#define value_t float
#define index_t int

struct diff2d_cpu : public diff2d<value_t, index_t>
{
    void benchmark()
    {
        print_bench();

        value_t **u = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **u_new = create_array2d<value_t, index_t>(side_size, side_size);

        initial_condition(u, u_new);

        loops = 0;
        auto start = std::chrono::high_resolution_clock::now();
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count()) < 1000.0 * benchtime)
        {
#pragma omp parallel for
            for (index_t i = 1; i < side_size - 1; i++)
            {
                for (index_t j = 1; j < side_size - 1; j++)
                {
                    u_new[i][j] = (1 - 4.0 * r) * u[i][j] // u_i_j
                                  + r * (u[i - 1][j]      // u_i-1_j
                                         + u[i + 1][j]    // u_i+1_j
                                         + u[i][j - 1]    // u_i_j-1
                                         + u[i][j + 1]);  // u_i_j+1
                }
            }

#pragma omp parallel for
            for (index_t j = 0; j < side_size; j++) // i = 0 y = all
                u_new[0][j] = 0.0;

#pragma omp parallel for
            for (index_t j = 0; j < side_size; j++) // i = end y = all
                u_new[side_size - 1][j] = 0.0;

#pragma omp parallel for
            for (index_t i = 1; i < side_size - 1; i++) // i = all except for two ends y = 0
                u_new[i][0] = 0.0;

#pragma omp parallel for
            for (index_t i = 1; i < side_size - 1; i++) // i = all except for two ends, y = end
                u_new[i][side_size - 1] = 0.0;

            // swap u and u_new
            value_t **tmp = u;
            u = u_new;
            u_new = tmp;

            loops++;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        duration = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        value_t t = t0 + delta_t * value_t(loops);
        test_result(u, t);
        print_performance();

        destroy_array2d<value_t, index_t>(u);
        destroy_array2d<value_t, index_t>(u_new);
    }

    diff2d_cpu(int narg, char **arg) : diff2d<value_t, index_t>(narg, arg) {}
};

int main(int narg, char **arg)
{
    diff2d_cpu test(narg, arg);
    test.benchmark();
}