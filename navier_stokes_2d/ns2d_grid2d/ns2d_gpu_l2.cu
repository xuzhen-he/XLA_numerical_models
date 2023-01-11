#include <chrono>
#include "array2d.h"
#include "cuda_helper.h"
#include "ns2d.h"

#define value_t double
#define index_t int

// constants
__constant__ value_t c_zero, c_two, c_half;

__global__ void predictor(index_t Nx, index_t Ny,
                          value_t *u, value_t *v, value_t *p,
                          value_t *u_star, value_t *v_star, value_t *p_star,
                          value_t dtdx, value_t dtdy,
                          value_t nu_dtdxx, value_t nu_dtdyy,
                          value_t c2_dtdx, value_t c2_dtdy,
                          value_t u0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = i * Ny + j;

    if (i < Nx && j < Ny)
    {
        if (i == 0) // i = 0 y = all
        {
            u_star[gid] = c_zero;
            v_star[gid] = c_zero;
            gid = gid + Ny;
            p_star[gid - Ny] = p[gid]                             //
                               - c2_dtdx * (u[gid + Ny] - u[gid]) //
                               - c2_dtdy * (v[gid + 1] - v[gid]);
        }
        else if (i == Nx - 1) // i = end y = all
        {
            u_star[gid] = c_zero;
            v_star[gid] = c_zero;
            gid = gid - Ny;
            p_star[gid + Ny] = p[gid]                             //
                               - c2_dtdx * (u[gid + Ny] - u[gid]) //
                               - c2_dtdy * (v[gid + 1] - v[gid]);
        }
        else
        {
            if (j == 0) // i = all except for two ends y = 0
            {
                u_star[gid] = c_zero;
                v_star[gid] = c_zero;
                gid = gid + 1;
                p_star[gid - 1] = p[gid]                             //
                                  - c2_dtdx * (u[gid + Ny] - u[gid]) //
                                  - c2_dtdy * (v[gid + 1] - v[gid]);
            }
            else if (j == Ny - 1) // i = all except for two ends y = end
            {
                u_star[gid] = u0;
                v_star[gid] = c_zero;
                gid = gid - 1;
                p_star[gid + 1] = p[gid]                             //
                                  - c2_dtdx * (u[gid + Ny] - u[gid]) //
                                  - c2_dtdy * (v[gid + 1] - v[gid]);
            }
            else
            {
                u_star[gid] = u[gid]                                                            //
                              - dtdx * (u[gid] * (u[gid + Ny] - u[gid]) + p[gid + Ny] - p[gid]) //
                              - dtdy * v[gid] * (u[gid + 1] - u[gid])                           //
                              + nu_dtdxx * (u[gid + Ny] - c_two * u[gid] + u[gid - Ny])           //
                              + nu_dtdyy * (u[gid + 1] - c_two * u[gid] + u[gid - 1]);
                v_star[gid] = v[gid]                                                          //
                              - dtdx * u[gid] * (v[gid + Ny] - v[gid])                        //
                              - dtdy * (v[gid] * (v[gid + 1] - v[gid]) + p[gid + 1] - p[gid]) //
                              + nu_dtdxx * (v[gid + Ny] - c_two * v[gid] + v[gid - Ny])         //
                              + nu_dtdyy * (v[gid + 1] - c_two * v[gid] + v[gid - 1]);
                p_star[gid] = p[gid]                             //
                              - c2_dtdx * (u[gid + Ny] - u[gid]) //
                              - c2_dtdy * (v[gid + 1] - v[gid]);
            }
        }
    }
}

__global__ void corrector(index_t Nx, index_t Ny,
                          value_t *u, value_t *v, value_t *p,
                          value_t *u_star, value_t *v_star, value_t *p_star,
                          value_t dtdx, value_t dtdy,
                          value_t nu_dtdxx, value_t nu_dtdyy,
                          value_t c2_dtdx, value_t c2_dtdy,
                          value_t u0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = i * Ny + j;

    if (i < Nx && j < Ny)
    {
        if (i == 0) // i = 0 y = all
        {
            u[gid] = c_zero;
            v[gid] = c_zero;
            gid = gid + Ny;
            value_t p_star2 = p_star[gid]                                  //
                              - c2_dtdx * (u_star[gid] - u_star[gid - Ny]) //
                              - c2_dtdy * (v_star[gid] - v_star[gid - 1]);
            p[gid - Ny] = c_half * (p[gid] + p_star2);
        }
        else if (i == Nx - 1) // i = end y = all
        {
            u[gid] = c_zero;
            v[gid] = c_zero;
            gid = gid - Ny;
            value_t p_star2 = p_star[gid]                                  //
                              - c2_dtdx * (u_star[gid] - u_star[gid - Ny]) //
                              - c2_dtdy * (v_star[gid] - v_star[gid - 1]);
            p[gid + Ny] = c_half * (p[gid] + p_star2);
        }
        else
        {
            if (j == 0) // i = all except for two ends y = 0
            {
                u[gid] = c_zero;
                v[gid] = c_zero;
                gid = gid + 1;
                value_t p_star2 = p_star[gid]                                  //
                                  - c2_dtdx * (u_star[gid] - u_star[gid - Ny]) //
                                  - c2_dtdy * (v_star[gid] - v_star[gid - 1]);
                p[gid - 1] = c_half * (p[gid] + p_star2);
            }
            else if (j == Ny - 1) // i = all except for two ends y = end
            {
                u[gid] = u0;
                v[gid] = c_zero;
                gid = gid - 1;
                value_t p_star2 = p_star[gid]                                  //
                                  - c2_dtdx * (u_star[gid] - u_star[gid - Ny]) //
                                  - c2_dtdy * (v_star[gid] - v_star[gid - 1]);
                p[gid + 1] = c_half * (p[gid] + p_star2);
            }
            else
            {
                value_t u_star2, v_star2, p_star2;
                u_star2 = u_star[gid]                                                                                //
                          - dtdx * (u_star[gid] * (u_star[gid] - u_star[gid - Ny]) + p_star[gid] - p_star[gid - Ny]) //
                          - dtdy * v_star[gid] * (u_star[gid] - u_star[gid - 1])                                     //
                          + nu_dtdxx * (u_star[gid + Ny] - c_two * u_star[gid] + u_star[gid - Ny])                     //
                          + nu_dtdyy * (u_star[gid + 1] - c_two * u_star[gid] + u_star[gid - 1]);
                v_star2 = v_star[gid]                                                                              //
                          - dtdx * u_star[gid] * (v_star[gid] - v_star[gid - Ny])                                  //
                          - dtdy * (v_star[gid] * (v_star[gid] - v_star[gid - 1]) + p_star[gid] - p_star[gid - 1]) //
                          + nu_dtdxx * (v_star[gid + Ny] - c_two * v_star[gid] + v_star[gid - Ny])                   //
                          + nu_dtdyy * (v_star[gid + 1] - c_two * v_star[gid] + v_star[gid - 1]);
                p_star2 = p_star[gid]                                  //
                          - c2_dtdx * (u_star[gid] - u_star[gid - Ny]) //
                          - c2_dtdy * (v_star[gid] - v_star[gid - 1]);
                u[gid] = c_half * (u[gid] + u_star2);
                v[gid] = c_half * (v[gid] + v_star2);
                p[gid] = c_half * (p[gid] + p_star2);
            }
        }
    }
}

inline void one_step(dim3 grid, dim3 block,
                     index_t Nx, index_t Ny,
                     value_t *d_u, value_t *d_v, value_t *d_p,
                     value_t *d_u_star, value_t *d_v_star, value_t *d_p_star,
                     value_t dtdx, value_t dtdy,
                     value_t nu_dtdxx, value_t nu_dtdyy,
                     value_t c2_dtdx, value_t c2_dtdy,
                     value_t u0)
{
    predictor<<<grid, block>>>(Nx, Ny,
                               d_u, d_v, d_p,
                               d_u_star, d_v_star, d_p_star,
                               dtdx, dtdy,
                               nu_dtdxx, nu_dtdyy,
                               c2_dtdx, c2_dtdy, u0);
    checkCudaErrorsAfterKernels;
    corrector<<<grid, block>>>(Nx, Ny,
                               d_u, d_v, d_p,
                               d_u_star, d_v_star, d_p_star,
                               dtdx, dtdy,
                               nu_dtdxx, nu_dtdyy,
                               c2_dtdx, c2_dtdy, u0);
    checkCudaErrorsAfterKernels;
}

struct ns2d_gpu : public ns2d<value_t, index_t>
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

        value_t *d_u, *d_v, *d_p;
        value_t *d_u_star, *d_v_star, *d_p_star;
        value_t *h_u = &u[0][0], *h_v = &v[0][0], *h_p = &p[0][0];
        checkCudaErrors(cudaMalloc(&d_u, total_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_v, total_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_p, total_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_u_star, total_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_v_star, total_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_p_star, total_size * sizeof(value_t)));

        checkCudaErrors(cudaMemcpy(d_u, h_u, total_size * sizeof(value_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_v, h_v, total_size * sizeof(value_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_p, h_p, total_size * sizeof(value_t), cudaMemcpyHostToDevice));
        value_t zero = 0.0;
        value_t two = 2.0;
        value_t half = 0.5;
        checkCudaErrors(cudaMemcpyToSymbol(c_zero, &zero, sizeof(value_t), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(c_two, &two, sizeof(value_t), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(c_half, &half, sizeof(value_t), 0, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        dim3 blockd3 = dim3(block0, block1, 1);
        dim3 grid = calc_grid2d(blockd3, side_size, side_size);
        std::cout << "  Block: " << blockd3.x << "(x) X " << blockd3.y << "(y)\n"
                  << "  Grid size: " << grid.x << "(x) X " << grid.y << "(y)\n\n";

        value_t e0 = get_u_increment(grid, blockd3,
                                     side_size, side_size,
                                     d_u, d_v, d_p,
                                     d_u_star, d_v_star, d_p_star,
                                     dtdx, dtdy,
                                     nu_dtdxx, nu_dtdyy,
                                     c2_dtdx, c2_dtdy,
                                     u0);

        loops = 0;
        auto startcpu = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaEventRecord(start));
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startcpu)
                    .count()) < 1000.0 * benchtime)
        // while (loops < 200000)
        {
            one_step(grid, blockd3,
                     side_size, side_size,
                     d_u, d_v, d_p,
                     d_u_star, d_v_star, d_p_star,
                     dtdx, dtdy,
                     nu_dtdxx, nu_dtdyy,
                     c2_dtdx, c2_dtdy, u0);

            loops++;
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float du = 0;
        checkCudaErrors(cudaEventElapsedTime(&du, start, stop));
        duration = 1.0e-3 * du;

        checkCudaErrors(cudaMemcpy(h_u, d_u, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_v, d_v, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_p, d_p, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));

        value_t ef = get_u_increment(grid, blockd3,
                                     side_size, side_size,
                                     d_u, d_v, d_p,
                                     d_u_star, d_v_star, d_p_star,
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
        checkCudaErrors(cudaFree(d_u));
        checkCudaErrors(cudaFree(d_v));
        checkCudaErrors(cudaFree(d_p));
        checkCudaErrors(cudaFree(d_u_star));
        checkCudaErrors(cudaFree(d_v_star));
        checkCudaErrors(cudaFree(d_p_star));
    }

    value_t get_u_increment(dim3 grid, dim3 block,
                            index_t Nx, index_t Ny,
                            value_t *d_u, value_t *d_v, value_t *d_p,
                            value_t *d_u_star, value_t *d_v_star, value_t *d_p_star,
                            value_t dtdx, value_t dtdy,
                            value_t nu_dtdxx, value_t nu_dtdyy,
                            value_t c2_dtdx, value_t c2_dtdy,
                            value_t u0)
    {
        value_t **h_u = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **u_inc = create_array2d<value_t, index_t>(side_size, side_size);

        checkCudaErrors(cudaMemcpy(h_u[0], d_u, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));
#pragma omp parallel for
        for (index_t i = 0; i < Nx; i++)
        {
            for (index_t j = 0; j < Ny; j++)
            {
                u_inc[i][j] = -h_u[i][j];
            }
        }

        one_step(grid, block,
                 side_size, side_size,
                 d_u, d_v, d_p,
                 d_u_star, d_v_star, d_p_star,
                 dtdx, dtdy,
                 nu_dtdxx, nu_dtdyy,
                 c2_dtdx, c2_dtdy, u0);

        checkCudaErrors(cudaMemcpy(h_u[0], d_u, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));
        value_t sum = 0.0;
        for (index_t i = 0; i < Nx; i++)
        {
            for (index_t j = 0; j < Ny; j++)
            {
                u_inc[i][j] += h_u[i][j];
                sum += u_inc[i][j] * u_inc[i][j];
            }
        }
        destroy_array2d<value_t, index_t>(u_inc);
        return sum;
    }

    ns2d_gpu(int narg, char **arg) : ns2d(narg, arg)
    {
    }
};

int main(int narg, char **arg)
{
    ns2d_gpu test(narg, arg);
    test.benchmark();
}