#include <chrono>
#include <string>
#include <omp.h>
#include <thread>
#include "array2d.h"
#include "mat_bench.h"

#define value_t float
#define index_t int

#define XPXPY 20

struct mat_xpxpy : public mat_bench<value_t, index_t>
{

        void benchmark()
        {
                print_bench();

#if XPXPY == 6
                std::cout << "\nSimulation info: 2d mat xpxpy 6\n";
#elif XPXPY == 12
                std::cout << "\nSimulation info: 2d mat xpxpy 12\n";
#elif XPXPY == 20
                std::cout << "\nSimulation info: 2d mat xpxpy 20\n";
#else
#endif

                value_t **x = create_array2d<value_t, index_t>(side_size, side_size);
                value_t **y = create_array2d<value_t, index_t>(side_size, side_size);

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
                                for (index_t j = 0; j < side_size; j++)
                                {
#if XPXPY == 6
                                        y[i][j] += x[i][j] - x[i][j] + x[i][j] - x[i][j] + x[i][j] - x[i][j];
#elif XPXPY == 12
                                        y[i][j] += x[i][j] - x[i][j] + x[i][j] - x[i][j] + x[i][j] - x[i][j] //
                                                   + x[i][j] - x[i][j] + x[i][j] - x[i][j] + x[i][j] - x[i][j];
#elif XPXPY == 20
                                        y[i][j] += x[i][j] - x[i][j] + x[i][j] - x[i][j] + x[i][j] - x[i][j]   //
                                                   + x[i][j] - x[i][j] + x[i][j] - x[i][j] + x[i][j] - x[i][j] //
                                                   + x[i][j] - x[i][j] + x[i][j] - x[i][j] + x[i][j] - x[i][j] //
                                                   + x[i][j] - x[i][j];
#else

#endif
                                }
                        }
                        loops++;
                }
                auto stop = std::chrono::high_resolution_clock::now();
                duration = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

                test_result(y, 0.0);
                print_performance();

                destroy_array2d<value_t, index_t>(x);
                destroy_array2d<value_t, index_t>(y);
        }

        mat_xpxpy(int narg, char **arg) : mat_bench<value_t, index_t>(narg, arg)
        {
                memory_transfer_per_loop = 3.0 * sizeof(value_t) * double(total_size) /
                                           (1024.0 * 1024.0 * 1024.0);
        }
};

int main(int narg, char **arg)
{
        mat_xpxpy test(narg, arg);
        test.benchmark();
}