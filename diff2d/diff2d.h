#include "bench.h"
#include "array2d.h"
#include "Kahan_summation.h"

#ifndef DIFF2D_H
#define DIFF2D_H

template <typename value_t, typename index_t>
struct diff2d : public bench2d<value_t, index_t>
{
    using bench<value_t, index_t>::total_size;
    using bench2d<value_t, index_t>::side_size;
    using bench<value_t, index_t>::memory_transfer_per_loop;

    value_t a;
    value_t r;
    value_t delta_x;
    value_t delta_t;

    value_t t0;

    diff2d(int narg, char **arg) : bench2d<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 2.0 * sizeof(value_t) * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);

        delta_x = 2.0 / (side_size - 1.0);
        a = 1.0;
        r = 0.2;
        delta_t = r * delta_x * delta_x / a;
        t0 = 0.001;

        std::cout << "\nSimulation info: 2d diffusion equation\n"
                  << "  Domain from -1 to 1\n"
                  << "  dx: " << delta_x << '\n'
                  << "  coefficient a: " << a << '\n'
                  << "  r = a*dt/dx: " << r << '\n'
                  << "  dt: " << delta_t << '\n'
                  << "  t0: " << t0 << '\n'
                  << "  u at centre initially: " << analytical(0.0, 0.0, t0) << '\n';
    }

    inline value_t analytical(value_t x, value_t y, value_t t)
    {
        return exp(-(x * x + y * y) / (4.0 * a * t)) / (4.0 * M_PI * a * t);
    }

    void initial_condition(value_t **u, value_t **u_new)
    {
#pragma omp parallel for
        for (index_t i = 0; i < side_size; i++)
        {
            for (index_t j = 0; j < side_size; j++)
            {
                value_t x = -1.0 + value_t(i) * delta_x;
                value_t y = -1.0 + value_t(j) * delta_x;
                u[i][j] = analytical(x, y, t0);
                u_new[i][j] = 0.0;
            }
        }
    }

    void test_result(value_t **u, value_t t)
    {
        value_t **e2 = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **ua2 = create_array2d<value_t, index_t>(side_size, side_size);

#pragma omp parallel for
        for (index_t i = 0; i < side_size; i++)
        {
            for (index_t j = 0; j < side_size; j++)
            {
                value_t x = -1.0 + value_t(i) * delta_x;
                value_t y = -1.0 + value_t(j) * delta_x;
                value_t u_analytical = analytical(x, y, t);
                ua2[i][j] = u_analytical * u_analytical;
                value_t error = u[i][j] - u_analytical;
                e2[i][j] = error * error;
            }
        }

        value_t sum_e2 = sqrt(Kahan_summation<value_t, index_t>(e2[0], total_size) / total_size);
        value_t sum_ua2 = sqrt(Kahan_summation<value_t, index_t>(ua2[0], total_size) / total_size);

        index_t mid = side_size / 2;
        value_t x_mid = -1.0 + value_t(mid) * delta_x;
        value_t ua_mid = analytical(x_mid, x_mid, t);

        std::cout << " \nCheck result\n"
                  << "  t: " << t << '\n'
                  << "  u at centre: " << u[mid][mid] << '\n'
                  << "  u analytical at centre: " << ua_mid << '\n'
                  << "  mean squared analytical: " << sum_ua2 << '\n'
                  << "  mean squared error: " << sum_e2 << '\n';

        destroy_array2d<value_t, index_t>(e2);
        destroy_array2d<value_t, index_t>(ua2);
    }
};

#endif