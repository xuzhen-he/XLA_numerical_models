#!/bin/bash

# sizes=(64  256  1024  4096  16384  65536  262144  1048576  4194304  16777216  67108864  268435456)

sizes=(4096  8192 16384  65536  262144 524288 1048576  4194304  16777216  67108864  268435456 536870912)
## now loop through the above array
for i in ${sizes[@]}
do
#    echo $i
    # bin/diff1d_cpu -size=$i
    # bin/diff1d_gpu_l2 -size=$i
    # bin/diff1d_gpu_shm -size=$i
    # python diff1d_stack.py --size $i
    # python diff1d_conv.py --size $i
    python diff1d_roll.py --size $i
   # or do whatever with individual element of the array
done