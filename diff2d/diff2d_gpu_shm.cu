#include <chrono>
#include "diff2d.h"
#include "cuda_helper.h"

#define value_t float
#define index_t int

// constants
__constant__ value_t c_zero, c_one, c_four;

__global__ void kernel(index_t Nx, index_t Ny, value_t r, value_t *u, value_t *u_new)
{
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    int gj = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = gi * Ny + gj;

    int ly = threadIdx.y;
    int lx = threadIdx.x;

    int slengthy = blockDim.y + 2;
    // int slengthx = blockDim.x + 2;
    int sx = threadIdx.x + 1;
    int sy = threadIdx.y + 1;
    int sid = sx * slengthy + sy;

    extern __shared__ value_t s_u[]; // (slengthy + 2) * (slengthx + 2)

    if (gi < Nx && gj < Ny)
    {
        s_u[sid] = u[gid];

        if (lx == 0 && gi != 0) // left
            s_u[sid - slengthy] = u[gid - Ny];

        if (lx == blockDim.x - 1 && gi != Nx - 1) // top
            s_u[sid + slengthy] = u[gid + Ny];

        if (ly == 0 && gj != 0)
        {
            s_u[sid - 1] = u[gid - 1]; // left
            if (lx == 0 && gi != 0)    // left bot corner
                s_u[sid - slengthy - 1] = u[gid - Ny - 1];
            if (lx == blockDim.x - 1 && gi != Nx - 1) // left top corner
                s_u[sid + slengthy - 1] = u[gid + Ny - 1];
        }

        if (ly == blockDim.y - 1 && gj != Ny - 1)
        {
            s_u[sid + 1] = u[gid + 1]; // right
            if (lx == 0 && gi != 0)    // right bot corner
                s_u[sid - slengthy + 1] = u[gid - Ny + 1];
            if (lx == blockDim.x - 1 && gi != Nx - 1) // righ top corner
                s_u[sid + slengthy + 1] = u[gid + Ny + 1];
        }
    }

    __syncthreads();

    if (gi < Nx && gj < Ny)
    {
        if (gi == 0) // i = 0 y = all
        {
            u_new[gid] = c_zero;
        }
        else if (gi == Nx - 1) // i = end y = all
        {
            u_new[gid] = c_zero;
        }
        else
        {
            if (gj == 0) // i = all except for two ends y = 0
            {
                u_new[gid] = c_zero;
            }
            else if (gj == Ny - 1) // i = all except for two ends y = end
            {
                u_new[gid] = c_zero;
            }
            else
            {
                u_new[gid] = (c_one - c_four * r) * s_u[sid] // u_i_j
                             + r * (s_u[sid - slengthy]      // u_i-1_j
                                    + s_u[sid + slengthy]    // u_i+1_j
                                    + s_u[sid - 1]           // u_i_j-1
                                    + s_u[sid + 1]);         // u_i_j+1
            }
        }
    }
}

struct diff2d_cuda_shm : public diff2d<value_t, index_t>
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

        // for(int i=0;i<total_size;i++)
        //     h_u[i] = value_t(i);

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
        int sm_memsize = (blockd3.x + 2) * (blockd3.y + 2) * sizeof(value_t);
        std::cout << "  Block: " << blockd3.x << "(x) X " << blockd3.y << "(y)\n"
                  << "  Grid size: " << grid.x << "(x) X " << grid.y << "(y)\n"
                  << "  Shared memory needed: " << sm_memsize << " Byte\n\n";

        loops = 0;
        auto startcpu = std::chrono::high_resolution_clock::now();
        checkCudaErrors(cudaEventRecord(start));
        while ((std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - startcpu)
                    .count()) < 1000.0 * benchtime)
        {
            kernel<<<grid, blockd3, sm_memsize>>>(side_size, side_size, r, d_u, d_u_new);
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
        // write_txt_array2d<value_t,index_t>(u,side_size,side_size,"test.txt");
        value_t t = delta_t * value_t(loops) + t0;
        test_result(u, t);
        print_performance();

        destroy_array2d<value_t, index_t>(u);
        destroy_array2d<value_t, index_t>(u_new);
        checkCudaErrors(cudaFree(d_u));
        checkCudaErrors(cudaFree(d_u_new));
    }

    diff2d_cuda_shm(int narg, char **arg) : diff2d(narg, arg) {}
};

int main(int narg, char **arg)
{
    check_cuda_device();
    diff2d_cuda_shm test(narg, arg);
    test.benchmark();
}