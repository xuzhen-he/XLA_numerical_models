#include "bench.h"

#ifndef NS2D_H
#define NS2D_H

template <typename value_t, typename index_t>
struct ns2d : public bench2d<value_t, index_t>
{
    using bench2d<value_t, index_t>::total_size;
    using bench2d<value_t, index_t>::side_size;
    using bench2d<value_t, index_t>::memory_transfer_per_loop;

    value_t dtdx;
    value_t dtdy;
    value_t nu_dtdxx;
    value_t nu_dtdyy;
    value_t c2_dtdx;
    value_t c2_dtdy;

    value_t u0;

    ns2d(int narg, char **arg) : bench2d<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 15.0 * sizeof(value_t) * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);

        value_t Re = 100.0;
        value_t Ma = 0.1;

        int i = 1;
        size_t found = 0;
        while (i < narg)
        {
            found = std::string(arg[i]).find("-re");
            if (found != std::string::npos)
            {
                Re = (value_t)atof(std::string(arg[i]).erase(0, found + 4).c_str());
            }

            found = std::string(arg[i]).find("-ma");
            if (found != std::string::npos)
            {
                Ma = (value_t)atof(std::string(arg[i]).erase(0, found + 4).c_str());
            }
            i++;
        }

        u0 = 1.0;
        value_t ll = 1.0;
        value_t nu = u0 * ll / Re;
        value_t sound_speed = u0 / Ma;

        value_t dx = 1.0 / (side_size - 1.0);
        value_t dy = 1.0 / (side_size - 1.0);
        value_t safe_factor_CFL = 0.2;
        value_t safe_factor_diffusion = 0.1;

        value_t dt_CFL = safe_factor_CFL * Ma * dx;
        value_t dt_Re = safe_factor_diffusion * Re * dx * dx;
        value_t dt = std::min(dt_CFL, dt_Re);

        dtdx = dt / dx;
        dtdy = dt / dy;
        value_t dtdxx = dt / (dx * dx);
        value_t dtdyy = dt / (dy * dy);
        nu_dtdxx = nu * dtdxx;
        nu_dtdyy = nu * dtdyy;
        c2_dtdx = sound_speed * sound_speed * dtdx;
        c2_dtdy = sound_speed * sound_speed * dtdy;

        std::cout << "\nSimulation info: 2d cavity flow\n"
                  << "  Domain x from 0 to 1, y from 0 to 1\n"
                  << "  Re: " << Re << '\n'
                  << "  Ma: " << Ma << '\n'
                  << "  dx: " << dx << '\n'
                  << "  dt: " << dt << '\n'
                  << "  lid velocity = " << u0 << '\n';
    }

    void initial_condition(value_t **u)
    {
        for (int i = 0; i < side_size; ++i)
            u[i][side_size - 1] = u0;
    }

    void test_result(value_t **u, value_t t)
    {
    }
};

#endif