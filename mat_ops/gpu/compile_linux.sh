# 
nvcc -arch=sm_61 -Xcompiler -fPIC,-Wall,-Wextra,-O3,-fopenmp,-fopt-info-vec,-march=native -I../.. -I.. -o bin/mat_xpxpy mat_xpxpy.cu