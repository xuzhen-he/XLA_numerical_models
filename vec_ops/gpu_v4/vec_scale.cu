#include <chrono>
#include "array2d.h"
#include "cuda_helper.h"
#include "vec_bench.h"

#define value_t float
#define index_t int

__global__ void kernel(index_t n, value_t scale, value_t *x)
{
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 4)
    {
        auto tmp = reinterpret_cast<float4 *>(x);
        tmp[i].x *= scale;
        tmp[i].y *= scale;
        tmp[i].z *= scale;
        tmp[i].w *= scale;
    }

    // in only one thread, process final elements (if there are any)
    index_t remainder = n % 4;
    if (i == n / 4 && remainder != 0)
    {
        while (remainder)
        {
            int i = n - remainder--;
            x[i] *= scale;
        }
    }
}

struct vec_copy : public vec_bench<value_t, index_t>
{
    void benchmark()
    {
        print_bench();

        std::cout << "\nSimulation info: 1d vec scale\n";

        value_t *x = new value_t[total_size];
        value_t doubleit = 2.0;
        value_t halfit = 0.5;

#pragma omp parallel for
        for (index_t j = 0; j < total_size; j++)
        {
            x[j] = 1.0;
        }

        value_t *d_x;
        checkCudaErrors(cudaMalloc(&d_x, total_size * sizeof(value_t)));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        checkCudaErrors(cudaMemcpy(d_x, x, total_size * sizeof(value_t), cudaMemcpyHostToDevice));

        dim3 blockd3 = dim3(block, 1, 1);
        dim3 grid = calc_grid1d(blockd3, total_size/4);
        std::cout << "  Block: " << blockd3.x << "(x) X " << blockd3.y << "(y)\n"
                  << "  Grid size: " << grid.x << "\n\n";

        loops = 0;
        auto startcpu = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaEventRecord(start));
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startcpu)
                    .count()) < 1000.0 * benchtime)
        {
            kernel<<<grid, block>>>(total_size, doubleit, d_x);
            kernel<<<grid, block>>>(total_size, halfit, d_x);
            checkCudaErrorsAfterKernels;
            loops++;
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float du = 0;
        checkCudaErrors(cudaEventElapsedTime(&du, start, stop));
        duration = 1.0e-3 * du;

        checkCudaErrors(cudaMemcpy(x, d_x, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));

        test_result(x, value_t(total_size));
        print_performance();

        delete[] x;
        checkCudaErrors(cudaFree(d_x));
    }

    vec_copy(int narg, char **arg) : vec_bench<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 2.0 * sizeof(value_t) * 2.0 * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);
    }
};

int main(int narg, char **arg)
{
    check_cuda_device();
    vec_copy test(narg, arg);
    test.benchmark();
}