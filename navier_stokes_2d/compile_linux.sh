#
# g++ -fPIC -Wall -Wextra -O3 -fopenmp -march=native -fopt-info-vec -I../.. -I.. ns2d_cpu.cpp -o bin/ns2d_cpu
nvcc -arch=sm_61 -Xcompiler -fPIC,-Wall,-Wextra,-O3,-fopenmp,-fopt-info-vec,-march=native -I../.. -I.. -o bin/ns2d_gpu_l2 ns2d_gpu_l2.cu  
