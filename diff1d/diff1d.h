#include "bench.h"
#include "Kahan_summation.h"

#ifndef DIFF1D_H
#define DIFF1D_H

template <typename value_t, typename index_t>
struct diff1d : public bench<value_t, index_t>
{
    using bench<value_t, index_t>::total_size;
    using bench<value_t, index_t>::memory_transfer_per_loop;

    value_t a;
    value_t r;
    value_t delta_x;
    value_t delta_t;

    diff1d(int narg, char **arg) : bench<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 2.0 * sizeof(value_t) * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);

        delta_x = 1.0 / (total_size - 1.0);
        a = 1.0;
        r = 0.2;
        delta_t = r * delta_x * delta_x / a;

        std::cout << "\nSimulation info: 1d diffussion eqation\n"
                  << "  x From 0 to 1\n"
                  << "  a: " << a << '\n'
                  << "  r = a*Dt/Dx: " << r << '\n';
    }

    void initial_condition(value_t *u, value_t *u_new)
    {
#pragma omp simd
        for (index_t j = 0; j < total_size; j++)
        {
            value_t x = value_t(j) * delta_x;
            u[j] = 6.0 * sin(M_PI * x);
            u_new[j] = 0.0;
        }
    }

    void test_result(value_t *u, value_t t)
    {
        value_t decay = exp(-a * M_PI * M_PI * t);

        index_t mid = total_size / 2;
        value_t midx = value_t(mid) * delta_x;
        value_t umidf_analytical = 6.0 * sin(M_PI * midx) * decay;
        value_t umidf = u[mid];

        value_t *ua2 = new value_t[total_size];
        value_t *e2 = new value_t[total_size];

#pragma omp simd
        for (index_t j = 0; j < total_size; j++)
        {
            value_t x = value_t(j) * delta_x;
            value_t u_analytical = 6.0 * sin(M_PI * x) * decay;
            ua2[j] = u_analytical * u_analytical;
            value_t e = u[j] - u_analytical;
            e2[j] = e * e;
        }

        value_t sum_ua2 = sqrt(Kahan_summation<value_t, index_t>(ua2, total_size)/total_size);
        value_t sum_e2 = sqrt(Kahan_summation<value_t, index_t>(e2, total_size)/total_size);

        std::cout << " \nCheck result\n"
                  << "  u at centre analyticallly: " << umidf_analytical << '\n'
                  << "  u at centre numerically: " << umidf << '\n'
                  << "  sum of u analyticall: " << sum_ua2 << '\n'
                  << "  sum of error: " << sum_e2 << '\n';

        delete[] ua2;
        delete[] e2;
    }
};

#endif