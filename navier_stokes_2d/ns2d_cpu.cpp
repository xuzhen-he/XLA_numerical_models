#include <chrono>
#include "array2d.h"
#include "Kahan_summation.h"
#include "ns2d.h"

#define value_t float
#define index_t int

inline void predictor(index_t Nx, index_t Ny,
                      value_t **u, value_t **v, value_t **p,
                      value_t **u_star, value_t **v_star, value_t **p_star,
                      value_t dtdx, value_t dtdy,
                      value_t nu_dtdxx, value_t nu_dtdyy,
                      value_t c2_dtdx, value_t c2_dtdy,
                      value_t u0)
{
#pragma omp parallel for
    for (index_t i = 1; i < Nx - 1; i++)
    {
        for (index_t j = 1; j < Nx - 1; j++)
        {
            u_star[i][j] = u[i][j]                                                              //
                           - dtdx * (u[i][j] * (u[i + 1][j] - u[i][j]) + p[i + 1][j] - p[i][j]) //
                           - dtdy * v[i][j] * (u[i][j + 1] - u[i][j])                           //
                           + nu_dtdxx * (u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j])             //
                           + nu_dtdyy * (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]);            //
            v_star[i][j] = v[i][j]                                                              //
                           - dtdx * u[i][j] * (v[i + 1][j] - v[i][j])                           //
                           - dtdy * (v[i][j] * (v[i][j + 1] - v[i][j]) + p[i][j + 1] - p[i][j]) //
                           + nu_dtdxx * (v[i + 1][j] - 2.0 * v[i][j] + v[i - 1][j])             //
                           + nu_dtdyy * (v[i][j + 1] - 2.0 * v[i][j] + v[i][j - 1]);
            p_star[i][j] = p[i][j]                             //
                           - c2_dtdx * (u[i + 1][j] - u[i][j]) //
                           - c2_dtdy * (v[i][j + 1] - v[i][j]);
        }
    }

#pragma omp parallel for
    for (index_t j = 1; j < Ny - 1; j++)
    {
        u_star[0][j] = 0.0;
        v_star[0][j] = 0.0;
        p_star[0][j] = p_star[1][j];

        u_star[Nx - 1][j] = 0.0;
        v_star[Nx - 1][j] = 0.0;
        p_star[Nx - 1][j] = p_star[Nx - 2][j];
    }

#pragma omp parallel for
    for (index_t i = 0; i < Nx; i++)
    {
        u_star[i][0] = 0.0;
        v_star[i][0] = 0.0;
        p_star[i][0] = p_star[i][1];

        u_star[i][Ny - 1] = u0;
        v_star[i][Ny - 1] = 0.0;
        p_star[i][Ny - 1] = p_star[i][Ny - 2];
    }
}

inline void corrector(index_t Nx, index_t Ny,
                      value_t **u, value_t **v, value_t **p,
                      value_t **u_star, value_t **v_star, value_t **p_star,
                      value_t dtdx, value_t dtdy,
                      value_t nu_dtdxx, value_t nu_dtdyy,
                      value_t c2_dtdx, value_t c2_dtdy,
                      value_t u0)
{
#pragma omp parallel for
    for (index_t i = 1; i < Nx - 1; i++)
    {
        for (index_t j = 1; j < Ny - 1; j++)
        {
            value_t u_star2, v_star2, p_star2;
            u_star2 = u_star[i][j] - dtdx * (u_star[i][j] * (u_star[i][j] - u_star[i - 1][j]) + p_star[i][j] - p_star[i - 1][j]) - dtdy * v_star[i][j] * (u_star[i][j] - u_star[i][j - 1]) + nu_dtdxx * (u_star[i + 1][j] - 2.0 * u_star[i][j] + u_star[i - 1][j]) + nu_dtdyy * (u_star[i][j + 1] - 2.0 * u_star[i][j] + u_star[i][j - 1]);
            v_star2 = v_star[i][j] - dtdx * u_star[i][j] * (v_star[i][j] - v_star[i - 1][j]) - dtdy * (v_star[i][j] * (v_star[i][j] - v_star[i][j - 1]) + p_star[i][j] - p_star[i][j - 1]) + nu_dtdxx * (v_star[i + 1][j] - 2.0 * v_star[i][j] + v_star[i - 1][j]) + nu_dtdyy * (v_star[i][j + 1] - 2.0 * v_star[i][j] + v_star[i][j - 1]);
            p_star2 = p_star[i][j] - c2_dtdx * (u_star[i][j] - u_star[i - 1][j]) - c2_dtdy * (v_star[i][j] - v_star[i][j - 1]);
            u[i][j] = 0.5 * (u[i][j] + u_star2);
            v[i][j] = 0.5 * (v[i][j] + v_star2);
            p[i][j] = 0.5 * (p[i][j] + p_star2);
        }
    }

#pragma omp parallel for
    for (index_t j = 1; j < Ny - 1; j++)
    {
        u[0][j] = 0.0;
        v[0][j] = 0.0;
        p[0][j] = p[1][j];

        u[Nx - 1][j] = 0.0;
        v[Nx - 1][j] = 0.0;
        p[Nx - 1][j] = p[Nx - 2][j];
    }

#pragma omp parallel for
    for (index_t i = 0; i < Nx; i++)
    {
        u[i][0] = 0.0;
        v[i][0] = 0.0;
        p[i][0] = p[i][1];

        u[i][Ny - 1] = u0;
        v[i][Ny - 1] = 0.0;
        p[i][Ny - 1] = p[i][Ny - 2];
    }
}

inline void one_step(index_t Nx, index_t Ny,
                     value_t **u, value_t **v, value_t **p,
                     value_t **u_star, value_t **v_star, value_t **p_star,
                     value_t dtdx, value_t dtdy,
                     value_t nu_dtdxx, value_t nu_dtdyy,
                     value_t c2_dtdx, value_t c2_dtdy,
                     value_t u0)
{
    predictor(Nx, Ny,
              u, v, p,
              u_star, v_star, p_star,
              dtdx, dtdy,
              nu_dtdxx, nu_dtdyy,
              c2_dtdx, c2_dtdy,
              u0);
    corrector(Nx, Ny,
              u, v, p,
              u_star, v_star, p_star,
              dtdx, dtdy,
              nu_dtdxx, nu_dtdyy,
              c2_dtdx, c2_dtdy,
              u0);
}

struct ns2d_cpu : public ns2d<value_t, index_t>
{
    void benchmark()
    {
        print_bench();

        value_t **u = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **v = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **p = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **u_star = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **v_star = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **p_star = create_array2d<value_t, index_t>(side_size, side_size);

        initial_condition(u);

        value_t e0 = get_u_increment(side_size, side_size,
                                     u, v, p,
                                     u_star, v_star, p_star,
                                     dtdx, dtdy,
                                     nu_dtdxx, nu_dtdyy,
                                     c2_dtdx, c2_dtdy,
                                     u0);

        loops = 0;
        auto start = std::chrono::high_resolution_clock::now();
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count()) < 1000.0 * benchtime)
        // while (loops < 200000)
        {
            one_step(side_size, side_size,
                     u, v, p,
                     u_star, v_star, p_star,
                     dtdx, dtdy,
                     nu_dtdxx, nu_dtdyy,
                     c2_dtdx, c2_dtdy,
                     u0);
            loops++;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        duration = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        value_t ef = get_u_increment(side_size, side_size,
                                     u, v, p,
                                     u_star, v_star, p_star,
                                     dtdx, dtdy,
                                     nu_dtdxx, nu_dtdyy,
                                     c2_dtdx, c2_dtdy,
                                     u0);

        std::cout << " \nCheck result\n"
                  << "  u incremtal initially: " << e0 << '\n'
                  << "  u incremtal initially: " << ef << '\n'
                  << "  ratio: " << ef / e0 << '\n';

        print_performance();
        // std::string fname = "test";
        // write_txt_array2d<value_t, index_t>(u, side_size, side_size, fname + "_u");
        // write_txt_array2d<value_t, index_t>(v, side_size, side_size, fname + "_v");
        // write_txt_array2d<value_t, index_t>(p, side_size, side_size, fname + "_p");

        destroy_array2d<value_t, index_t>(u);
        destroy_array2d<value_t, index_t>(v);
        destroy_array2d<value_t, index_t>(p);
        destroy_array2d<value_t, index_t>(u_star);
        destroy_array2d<value_t, index_t>(v_star);
        destroy_array2d<value_t, index_t>(p_star);
    }

    value_t get_u_increment(index_t Nx, index_t Ny,
                            value_t **u, value_t **v, value_t **p,
                            value_t **u_star, value_t **v_star, value_t **p_star,
                            value_t dtdx, value_t dtdy,
                            value_t nu_dtdxx, value_t nu_dtdyy,
                            value_t c2_dtdx, value_t c2_dtdy,
                            value_t u0)
    {
        value_t **u_inc = create_array2d<value_t, index_t>(side_size, side_size);
#pragma omp parallel for
        for (index_t i = 0; i < Nx; i++)
        {
            for (index_t j = 0; j < Ny; j++)
            {
                u_inc[i][j] = -u[i][j];
            }
        }

        one_step(Nx, Ny,
                 u, v, p,
                 u_star, v_star, p_star,
                 dtdx, dtdy,
                 nu_dtdxx, nu_dtdyy,
                 c2_dtdx, c2_dtdy,
                 u0);

        value_t sum = 0.0;
        for (index_t i = 0; i < Nx; i++)
        {
            for (index_t j = 0; j < Ny; j++)
            {
                u_inc[i][j] += u[i][j];
                sum += u_inc[i][j] * u_inc[i][j];
            }
        }
        destroy_array2d<value_t, index_t>(u_inc);
        return sum;
    }

    ns2d_cpu(int narg, char **arg) : ns2d(narg, arg)
    {
    }
};

int main(int narg, char **arg)
{
    ns2d_cpu test(narg, arg);
    test.benchmark();
}