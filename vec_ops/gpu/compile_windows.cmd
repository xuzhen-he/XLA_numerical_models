@call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

@REM nvcc -arch=sm_61 -Xcompiler "/openmp:experimental" -o bin/vec_scale_gpu vec_scale.cu
nvcc -ptx vec_axpy.cu

pause