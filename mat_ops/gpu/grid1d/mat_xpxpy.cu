#include <chrono>
#include "array2d.h"
#include "cuda_helper.h"
#include "mat_bench.h"

#define value_t float
#define index_t int

#define XPXPY 20

__global__ void kernel(index_t Nx, index_t Ny, value_t *x, value_t *y)
{
    // grid moves along last index first
    // int N_grid_j = (Ny + blockDim.y - 1) / blockDim.y;
    // int grid_i = blockIdx.x / N_grid_j;
    // int grid_j = blockIdx.x - grid_i * N_grid_j;

    // grid moves along first index first
    int N_grid_i = (Nx + blockDim.x - 1) / blockDim.x;
    int grid_j = blockIdx.x / N_grid_i;
    int grid_i = blockIdx.x - grid_j * N_grid_i;

    int i = grid_i * blockDim.x + threadIdx.x;
    int j = grid_j * blockDim.y + threadIdx.y;
    int gid = i * Ny + j;
    if (i < Nx && j < Ny)
    {
#if XPXPY == 6
        y[gid] += x[gid] - x[gid] + x[gid] - x[gid] + x[gid] - x[gid];
#elif XPXPY == 12
        y[gid] += x[gid] - x[gid] + x[gid] - x[gid] + x[gid] - x[gid] //
                  + x[gid] - x[gid] + x[gid] - x[gid] + x[gid] - x[gid];
#elif XPXPY == 20
        y[gid] += x[gid] - x[gid] + x[gid] - x[gid] + x[gid] - x[gid]   //
                  + x[gid] - x[gid] + x[gid] - x[gid] + x[gid] - x[gid] //
                  + x[gid] - x[gid] + x[gid] - x[gid] + x[gid] - x[gid] //
                  + x[gid] - x[gid];
#else

#endif
    }
}

struct mat_copy : public mat_bench<value_t, index_t>
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

        value_t *d_x, *d_y;
        value_t *h_x = x[0], *h_y = y[0];
        checkCudaErrors(cudaMalloc(&d_x, total_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_y, total_size * sizeof(value_t)));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        checkCudaErrors(cudaMemcpy(d_x, h_x, total_size * sizeof(value_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y, h_y, total_size * sizeof(value_t), cudaMemcpyHostToDevice));

        dim3 blockd3 = dim3(block0, block1, 1);
        dim3 grid = calc_grid2d2(blockd3, side_size, side_size);
        std::cout << "  Block: " << blockd3.x << "(x) X " << blockd3.y << "(y)\n"
                  << "  Grid size: " << grid.x << "\n\n";

        loops = 0;
        auto startcpu = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaEventRecord(start));
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startcpu)
                    .count()) < 1000.0 * benchtime)
        {
            kernel<<<grid, blockd3>>>(side_size, side_size, d_x, d_y);
            checkCudaErrorsAfterKernels;
            loops++;
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float du = 0;
        checkCudaErrors(cudaEventElapsedTime(&du, start, stop));
        duration = 1.0e-3 * du;

        checkCudaErrors(cudaMemcpy(h_y, d_y, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));

        test_result(y, 0.0);
        print_performance();

        delete[] x;
        delete[] y;
        checkCudaErrors(cudaFree(d_x));
        checkCudaErrors(cudaFree(d_y));
    }

    mat_copy(int narg, char **arg) : mat_bench<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 3.0 * sizeof(value_t) * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);
    }
};

int main(int narg, char **arg)
{
    check_cuda_device();
    mat_copy test(narg, arg);
    test.benchmark();
}