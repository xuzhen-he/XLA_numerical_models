@echo off
setlocal EnableDelayedExpansion

set size=64 256 1024
set loops=1000000 1000000 100000

@REM set size=4096 16384 65536
@REM set loops=1000000 1000000 100000

@REM set size=262144 1048576 4194304
@REM set loops=10000 10000 1000

@REM set size=16777216 67108864 134217728
@REM set loops=1000 100 100

rem Convert the lists into arrays
set i=0
for %%i in (%size%) do (
   set /A i+=1
   set "size[!i!]=%%i"
)

@REM for /l %%i in (1,1,%i%) do (
@REM    echo !size[%%i]!
@REM )

set i=0
for %%i in (%loops%) do (
   set /A i+=1
   set "loops[!i!]=%%i"
)

call cd bin

for /l %%i in (1,1,%i%) do (
   call vec_scale_gpu.exe -size=!size[%%i]! -loops=!loops[%%i]!"
)

pause