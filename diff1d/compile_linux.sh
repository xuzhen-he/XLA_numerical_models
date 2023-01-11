#
# g++ -fPIC -Wall -Wextra -O3 -fopenmp -march=native -fopt-info-vec -I../.. -I.. diff1d_cpu.cpp -o bin/diff1d_cpu
nvcc -arch=sm_61 -Xcompiler -fPIC,-Wall,-Wextra,-O3,-fopenmp,-fopt-info-vec,-march=native -I../.. -I.. -o bin/diff1d_gpu_shm diff1d_gpu_shm.cu  
