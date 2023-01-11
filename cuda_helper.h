#include <iostream>

#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

// #define checkCudaErrors(ans) {gpuAssert((ans), __FILE__, __LINE__);}
// #define checkCudaErrorsAfterKernels     checkCudaErrors(cudaPeekAtLastError()); checkCudaErrors(cudaDeviceSynchronize());
#define checkCudaErrors(ans) (ans)
#define checkCudaErrorsAfterKernels

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(code) << " " << file << " " << line;
        if (abort)
            exit(code);
    }
}

inline void check_cuda_device()
{
    // check CUDA device
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        std::cout << "There are no available device(s) that support CUDA\n";
    }
    else
    {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)\n";
    }
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    checkCudaErrors(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    std::cout << "Device " << dev << '\t' << deviceProp.name << '\n';
    // Console log
    checkCudaErrors(cudaDriverGetVersion(&driverVersion));
    checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));
    std::cout << "  CUDA Driver Version / Runtime Version\t"
              << driverVersion / 1000 << '.' << (driverVersion % 100) / 10 << '/'
              << runtimeVersion / 1000 << '.' << (runtimeVersion % 100) / 10 << '\n';
    std::cout << "  CUDA Capability Major/Minor version number:\t"
              << deviceProp.major << '.' << deviceProp.minor << "\n\n";
}

dim3 calc_grid1d(dim3 block, int size)
{
    int total_grid = (size + block.x - 1) / block.x;
    dim3 grid = dim3(total_grid, 1, 1);
    return grid;
}

dim3 calc_grid2d(dim3 block, int size0, int size1)
{
    int N_grid0 = (size0 + block.x - 1) / block.x;
    int N_grid1 = (size1 + block.y - 1) / block.y;
    dim3 grid = dim3(N_grid0, N_grid1, 1);
    return grid;
}


dim3 calc_grid2d2(dim3 block, int size0, int size1)
{
    int N_grid0 = (size0 + block.x - 1) / block.x;
    int N_grid1 = (size1 + block.y - 1) / block.y;
    int total_grid = N_grid0 * N_grid1;
    dim3 grid = dim3(total_grid, 1, 1);
    return grid;
}



#endif