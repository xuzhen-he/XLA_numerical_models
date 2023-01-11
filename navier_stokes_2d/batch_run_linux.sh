#!/bin/bash

# sizes=(64  256  1024  4096  16384  65536  262144  1048576  4194304  16777216  67108864  268435456)
sizes=(4096  8192 16384  65536  262144 524288 1048576  4194304  16777216  67108864  268435456 536870912)
## now loop through the above array
for i in ${sizes[@]}
do
   # echo $i
   #  bin/ns2d_cpu -size=$i
   #  bin/ns2d_gpu_l2 -size=$i
   python ns2d_jax_stack1.py --size $i
   # or do whatever with individual element of the array
done