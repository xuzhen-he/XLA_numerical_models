#include <chrono>
#include <string>
#include "vec_bench.h"
#include "cuda_helper.h"

__global__ void scale(index_t n, value_t scale, value_t *x)
{
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 2)
    {
        auto tmp = reinterpret_cast<double2 *>(x);
        tmp[i].x *= scale;
        tmp[i].y *= scale;
    }

    // in only one thread, process final elements (if there are any)
    index_t remainder = n % 2;
    if (i == n / 2 && remainder != 0)
    {
        while (remainder)
        {
            int i = n - remainder--;
            x[i] *= scale;
        }
    }
}

struct vec_scale_cuda : public vec_bench
{
    void benchmark()
    {
        memory_transfer_per_loop = 2.0 * sizeof(value_t) * double(vec_size) * 2.0 /
                                   (1024.0 * 1024.0 * 1024.0);
        
        dim3 block = dim3(block_x, 1, 1);
        dim3 grid = dim3((vec_size/2 + block.x - 1) / block.x, 1, 1);
        std::cout << "  Block: " << block.x << "(x) X " << block.y << "(y)\n"
                  << "  Grid: " << grid.x << "(x) X " << grid.y << "(y)\n\n";

        value_t *x = new value_t[vec_size];
        value_t doubleit = 2.0;
        value_t halfit = 0.5;

        for (index_t j = 0; j < vec_size; j++)
        {
            x[j] = 1.0;
        }

        value_t *d_x;
        checkCudaErrors(cudaMalloc(&d_x, vec_size * sizeof(value_t)));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        checkCudaErrors(cudaMemcpy(d_x, x, vec_size * sizeof(value_t), cudaMemcpyHostToDevice));

        index_t loops = 0;
        auto startcpu = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaEventRecord(start));
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startcpu)
                    .count()) < 1000.0 * benchtime)
        {
            scale<<<grid, block>>>(vec_size, doubleit, d_x);
            scale<<<grid, block>>>(vec_size, halfit, d_x);
            checkCudaErrorsAfterKernels
                loops++;
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float duration = 0;
        checkCudaErrors(cudaEventElapsedTime(&duration, start, stop));

        checkCudaErrors(cudaMemcpy(x, d_x, vec_size * sizeof(value_t), cudaMemcpyDeviceToHost));

        test_result(x, vec_size, value_t(vec_size));
        std::cout << "  y[0] " << x[0] << '\n';
        print_performance(double(loops), 1.0e-3 * duration);

        delete[] x;
        checkCudaErrors(cudaFree(d_x));
    }

    vec_scale_cuda(int narg, char **arg) : vec_bench(narg, arg) {}
};

int main(int narg, char **arg)
{
    check_cuda_device();
    vec_scale_cuda test(narg, arg);
    test.benchmark();
}