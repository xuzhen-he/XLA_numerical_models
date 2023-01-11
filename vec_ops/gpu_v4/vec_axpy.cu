#include <chrono>
#include "array2d.h"
#include "cuda_helper.h"
#include "vec_bench.h"

#define value_t float
#define index_t int

__global__ void kernel(index_t n, value_t scale, value_t *x, value_t *y)
{
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 4)
    {
        auto tmp_y = reinterpret_cast<float4 *>(y);
        auto tmp_x = reinterpret_cast<float4 *>(x);
        tmp_y[i].x += scale * tmp_x[i].x;
        tmp_y[i].y += scale * tmp_x[i].x;
        tmp_y[i].z += scale * tmp_x[i].x;
        tmp_y[i].w += scale * tmp_x[i].x;
    }

    // in only one thread, process final elements (if there are any)
    index_t remainder = n % 4;
    if (i == n / 4 && remainder != 0)
    {
        while (remainder)
        {
            int i = n - remainder--;
            y[i] += scale * x[i];
        }
    }
}

struct vec_copy : public vec_bench<value_t, index_t>
{
    void benchmark()
    {
        print_bench();

        std::cout << "\nSimulation info: 1d vec axpy\n";

        value_t *x = new value_t[total_size];
        value_t *y = new value_t[total_size];
        value_t a = 1.0;

#pragma omp parallel for
        for (index_t j = 0; j < total_size; j++)
        {
            x[j] = 1.0;
            y[j] = 0.0;
        }

        value_t *d_x, *d_y;
        checkCudaErrors(cudaMalloc(&d_x, total_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_y, total_size * sizeof(value_t)));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        checkCudaErrors(cudaMemcpy(d_x, x, total_size * sizeof(value_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y, y, total_size * sizeof(value_t), cudaMemcpyHostToDevice));

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
            kernel<<<grid, blockd3>>>(total_size, a, d_x, d_y);
            checkCudaErrorsAfterKernels;
            loops++;
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float du = 0;
        checkCudaErrors(cudaEventElapsedTime(&du, start, stop));
        duration = 1.0e-3 * du;

        checkCudaErrors(cudaMemcpy(y, d_y, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));

        value_t target = (value_t)loops * (value_t)total_size;
        test_result(y, target);
        print_performance();

        delete[] x;
        delete[] y;
        checkCudaErrors(cudaFree(d_x));
        checkCudaErrors(cudaFree(d_y));
    }

    vec_copy(int narg, char **arg) : vec_bench<value_t, index_t>(narg, arg)
    {
        memory_transfer_per_loop = 3.0 * sizeof(value_t) * double(total_size) /
                                   (1024.0 * 1024.0 * 1024.0);
    }
};

int main(int narg, char **arg)
{
    check_cuda_device();
    vec_copy test(narg, arg);
    test.benchmark();
}