@call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

rem :experimental 
cl vec_copy.cpp -o "bin/vec_copy_omp.exe"   /EHsc /fp:precise /TP /MT /O2  /arch:AVX /openmp:experimental

del *.obj

pause
