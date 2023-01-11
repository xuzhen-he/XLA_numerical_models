#include <chrono>
#include "array2d.h"
#include "cuda_helper.h"
#include "mat_bench.h"

#define value_t float
#define index_t int

__global__ void kernel(index_t Nx, index_t Ny, value_t *x, value_t *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = i * Ny + j;
    if (i < Nx && j < Ny)
        y[gid] = x[gid];
}

struct mat_copy : public mat_bench<value_t, index_t>
{
    void benchmark()
    {
        print_bench();

        std::cout << "\nSimulation info: 2d mat copy\n";

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

        test_result(y, value_t(total_size));
        print_performance();

        delete[] x;
        delete[] y;
        checkCudaErrors(cudaFree(d_x));
        checkCudaErrors(cudaFree(d_y));
    }

    mat_copy(int narg, char **arg) : mat_bench<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 2.0 * sizeof(value_t) * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);
    }
};

int main(int narg, char **arg)
{
    check_cuda_device();
    mat_copy test(narg, arg);
    test.benchmark();
}