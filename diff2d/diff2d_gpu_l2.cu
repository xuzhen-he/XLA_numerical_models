#include <chrono>
#include "diff2d.h"
#include "cuda_helper.h"

#define value_t float
#define index_t int

// constants
__constant__ value_t c_zero, c_one, c_four;

__global__ void kernel(index_t Nx, index_t Ny, value_t r, value_t *u, value_t *u_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = i * Ny + j;

    if (i < Nx && j < Ny)
    {
        if (i == 0) // i = 0 y = all
        {
            u_new[gid] = c_zero;
        }
        else if (i == Nx - 1) // i = end y = all
        {
            u_new[gid] = c_zero;
        }
        else
        {
            if (j == 0) // i = all except for two ends y = 0
            {
                u_new[gid] = c_zero;
            }
            else if (j == Ny - 1) // i = all except for two ends y = end
            {
                u_new[gid] = c_zero;
            }
            else
            {
                u_new[gid] = (c_one - c_four * r) * u[gid] // u_i_j
                             + r * (u[gid - Ny]            // u_i-1_j
                                    + u[gid + Ny]          // u_i+1_j
                                    + u[gid - 1]           // u_i_j-1
                                    + u[gid + 1]);         // u_i_j+1
            }
        }
    }
}

struct diff2d_cuda_l2 : public diff2d<value_t, index_t>
{
    void benchmark()
    {
        print_bench();

        value_t **u = create_array2d<value_t, index_t>(side_size, side_size);
        value_t **u_new = create_array2d<value_t, index_t>(side_size, side_size);

        initial_condition(u, u_new);

        value_t *d_u, *d_u_new;
        value_t *h_u = &u[0][0], *h_u_new = &u_new[0][0];
        checkCudaErrors(cudaMalloc(&d_u, total_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_u_new, total_size * sizeof(value_t)));

        checkCudaErrors(cudaMemcpy(d_u, h_u, total_size * sizeof(value_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_u_new, h_u_new, total_size * sizeof(value_t), cudaMemcpyHostToDevice));
        value_t zero = 0.0;
        value_t one = 1.0;
        value_t four = 4.0;
        checkCudaErrors(cudaMemcpyToSymbol(c_zero, &zero, sizeof(value_t), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(c_one, &one, sizeof(value_t), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(c_four, &four, sizeof(value_t), 0, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        dim3 blockd3 = dim3(block0, block1, 1);
        dim3 grid = calc_grid2d(blockd3, side_size, side_size);
        std::cout << "  Block: " << blockd3.x << "(x) X " << blockd3.y << "(y)\n"
                  << "  Grid size: " << grid.x << "(x) X " << grid.y << "(y)\n\n";

        loops = 0;
        auto startcpu = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaEventRecord(start));
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startcpu)
                    .count()) < 1000.0 * benchtime)
        {
            kernel<<<grid, blockd3>>>(side_size, side_size, r, d_u, d_u_new);
            checkCudaErrorsAfterKernels;

            // swap u and u_new
            value_t *tmp = d_u;
            d_u = d_u_new;
            d_u_new = tmp;
            loops++;
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float du = 0;
        checkCudaErrors(cudaEventElapsedTime(&du, start, stop));
        duration = 1.0e-3 * du;

        checkCudaErrors(cudaMemcpy(h_u, d_u, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));

        value_t t = delta_t * value_t(loops) + t0;
        test_result(u, t);
        print_performance();

        destroy_array2d<value_t, index_t>(u);
        destroy_array2d<value_t, index_t>(u_new);
        checkCudaErrors(cudaFree(d_u));
        checkCudaErrors(cudaFree(d_u_new));
    }

    diff2d_cuda_l2(int narg, char **arg) : diff2d(narg, arg) {}
};

int main(int narg, char **arg)
{
    check_cuda_device();
    diff2d_cuda_l2 test(narg, arg);
    test.benchmark();
}