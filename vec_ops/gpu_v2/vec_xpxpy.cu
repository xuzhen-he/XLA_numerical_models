#include <chrono>
#include <string>
#include "vec_bench.h"
#include "cuda_helper.h"

__global__ void xpxpy(index_t n, value_t *x, value_t *y)
{
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 2)
    {
        auto tmp_y = reinterpret_cast<double2 *>(y);
        auto tmp_x = reinterpret_cast<double2 *>(x);
        tmp_y[i].x += tmp_x[i].x + tmp_x[i].x - tmp_x[i].x + tmp_x[i].x - tmp_x[i].x;
            //  + tmp_x[i].x - tmp_x[i].x + tmp_x[i].x - tmp_x[i].x + tmp_x[i].x - tmp_x[i].x;
            //  + tmp_x[i].x - tmp_x[i].x + tmp_x[i].x - tmp_x[i].x + tmp_x[i].x- tmp_x[i].x//;
            //  + tmp_x[i].x - tmp_x[i].x + tmp_x[i].x - tmp_x[i].x;

        tmp_y[i].y += tmp_x[i].y + tmp_x[i].y - tmp_x[i].y + tmp_x[i].y - tmp_x[i].y;
            //  + tmp_x[i].y - tmp_x[i].y + tmp_x[i].y - tmp_x[i].y + tmp_x[i].y - tmp_x[i].y;
            //  + tmp_x[i].y - tmp_x[i].y + tmp_x[i].y - tmp_x[i].y + tmp_x[i].y- tmp_x[i].y//;
            //  + tmp_x[i].y - tmp_x[i].y + tmp_x[i].y - tmp_x[i].y;
    }

    // in only one thread, process final elements (if there are any)
    index_t remainder = n % 2;
    if (i == n / 2 && remainder != 0)
    {
        while (remainder)
        {
            int i = n - remainder--;
            y[i] += x[i] + x[i] - x[i] + x[i] - x[i];
            //  + x[i] - x[i] + x[i] - x[i] + x[i] - x[i];
            //  + x[i] - x[i] + x[i] - x[i] + x[i]- x[i]//;
            //  + x[i] - x[i] + x[i] - x[i];
        }
    }
}

struct vec_xpxpy_cuda : public vec_bench
{
    void benchmark()
    {
        memory_transfer_per_loop = 3.0 * sizeof(value_t) * double(vec_size) /
                                   (1024.0 * 1024.0 * 1024.0);

        dim3 block = dim3(block_x, 1, 1);
        dim3 grid = dim3((vec_size/2 + block.x - 1) / block.x, 1, 1);
        std::cout << "  Block: " << block.x << "(x) X " << block.y << "(y)\n"
                  << "  Grid: " << grid.x << "(x) X " << grid.y << "(y)\n\n";

        value_t *x = new value_t[vec_size];
        value_t *y = new value_t[vec_size];

        for (index_t j = 0; j < vec_size; j++)
        {
            x[j] = 1.0;
            y[j] = 0.0;
        }

        value_t *d_x, *d_y;
        checkCudaErrors(cudaMalloc(&d_x, vec_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_y, vec_size * sizeof(value_t)));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        checkCudaErrors(cudaMemcpy(d_x, x, vec_size * sizeof(value_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y, y, vec_size * sizeof(value_t), cudaMemcpyHostToDevice));

        index_t loops = 0;
        auto startcpu = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaEventRecord(start));
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startcpu)
                    .count()) < 1000.0 * benchtime)
        {
            xpxpy<<<grid, block>>>(vec_size, d_x, d_y);
            checkCudaErrorsAfterKernels
                loops++;
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float duration = 0;
        checkCudaErrors(cudaEventElapsedTime(&duration, start, stop));

        checkCudaErrors(cudaMemcpy(y, d_y, vec_size * sizeof(value_t), cudaMemcpyDeviceToHost));

        value_t target = (value_t)loops * (value_t)vec_size;
        test_result(y, vec_size, target);
        std::cout << "  y[0] " << y[0] << '\n';
        print_performance(double(loops), 1.0e-3 * duration);

        delete[] x;
        delete[] y;
        checkCudaErrors(cudaFree(d_x));
        checkCudaErrors(cudaFree(d_y));
    }

    vec_xpxpy_cuda(int narg, char **arg) : vec_bench(narg, arg) {}
};

int main(int narg, char **arg)
{
    check_cuda_device();
    vec_xpxpy_cuda test(narg, arg);
    test.benchmark();
}