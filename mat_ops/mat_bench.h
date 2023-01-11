#include "bench.h"
#include "Kahan_summation.h"

#ifndef MAT_BENCH_H
#define MAT_BENCH_H

template <typename value_t, typename index_t>
struct mat_bench : public bench2d<value_t, index_t>
{
    using bench<value_t, index_t>::total_size;
    using bench<value_t, index_t>::memory_transfer_per_loop;

    mat_bench(int narg, char **arg) : bench2d<value_t, index_t>(narg, arg)
    {
    }

    void test_result(value_t **y, value_t target)
    {
        // Kahan summation algorithm to improve accuracy
        value_t sum = Kahan_summation<value_t, index_t>(*y, total_size);
        bool pass;
        if (target == 0.0)
            pass = abs(sum - target) < 1.0e-10;
        else
            pass = abs(sum - target) / target < 1.0e-10;
        std::cout << " \nCheck result\n"
                  << "  y[0] " << y[0][0] << '\n'
                  << "  sum y " << sum << '\n'
                  << (pass ? "  equal\n" : "  not equal\n")
                  << "  target " << target << '\n';
    }
};

#endif