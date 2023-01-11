#include <chrono>
#include <cmath>
#include <string>
#include <omp.h>
#include <thread>
#include "diff1d.h"
#include "cuda_helper.h"

#define value_t float
#define index_t int

// constants
__constant__ value_t c_zero, c_one, c_two;

__global__ void kernel(index_t n, value_t r, value_t *u, value_t *u_new)
{
  index_t lid = threadIdx.x;
  index_t sid = lid + 1;
  index_t width = blockDim.x;
  index_t gid = blockIdx.x * width + threadIdx.x;

  extern __shared__ value_t s_u[]; // width + 2

  if (gid < n)
  {
    s_u[sid] = u[gid];

    if (lid == 0 && gid != 0)
      s_u[sid - 1] = u[gid - 1];

    if (lid == width - 1 && gid != n - 1)
      s_u[sid + 1] = u[gid + 1];
  }

  __syncthreads();

  if (gid < n)
  {
    if (gid == 0)
      u_new[gid] = c_zero;
    else if (gid == n - 1)
      u_new[gid] = c_zero;
    else
      u_new[gid] = (c_one - c_two * r) * s_u[sid] + r * (s_u[sid - 1] + s_u[sid + 1]);
  }
}

struct diff1d_l2 : public diff1d<value_t, index_t>
{
    void benchmark()
    {
        print_bench();

        value_t *u = new value_t[total_size];
        value_t *u_new = new value_t[total_size];

        initial_condition(u, u_new);

        value_t *d_u, *d_u_new;
        checkCudaErrors(cudaMalloc(&d_u, total_size * sizeof(value_t)));
        checkCudaErrors(cudaMalloc(&d_u_new, total_size * sizeof(value_t)));

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        checkCudaErrors(cudaMemcpy(d_u, u, total_size * sizeof(value_t), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_u_new, u_new, total_size * sizeof(value_t), cudaMemcpyHostToDevice));

        value_t zero = 0.0;
        value_t one = 1.0;
        value_t two = 2.0;
        checkCudaErrors(cudaMemcpyToSymbol(c_zero, &zero, sizeof(value_t), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(c_one, &one, sizeof(value_t), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(c_two, &two, sizeof(value_t), 0, cudaMemcpyHostToDevice));

        dim3 blockd3 = dim3(block, 1, 1);
        dim3 grid = calc_grid1d(blockd3, total_size);
        std::cout << "  Block: " << blockd3.x << "(x) X " << blockd3.y << "(y)\n"
                  << "  Grid size: " << grid.x << "\n";
        int sm_memsize = (blockd3.x + 2) * sizeof(value_t);
        std::cout << "  Shared memory needed: " << sm_memsize << " Byte\n\n";

        loops = 0;
        auto startcpu = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaEventRecord(start));
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startcpu)
                    .count()) < 1000.0 * benchtime)
        {
            kernel<<<grid, block, sm_memsize>>>(total_size, r, d_u, d_u_new);
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

        checkCudaErrors(cudaMemcpy(u, d_u, total_size * sizeof(value_t), cudaMemcpyDeviceToHost));

        value_t t = delta_t * value_t(loops);
        test_result(u, t);
        print_performance();

        delete[] u;
        delete[] u_new;
        checkCudaErrors(cudaFree(d_u));
        checkCudaErrors(cudaFree(d_u_new));
    }

    diff1d_l2(int narg, char **arg) : diff1d<value_t, index_t>(narg, arg) {}
};

int main(int narg, char **arg)
{
    check_cuda_device();
    diff1d_l2 test(narg, arg);
    test.benchmark();
}