#!/bin/bash

sizes=(4096  8192 16384  65536  262144 524288 1048576  4194304  16777216  67108864  268435456 536870912)
## now loop through the above array
for i in ${sizes[@]}
do
#    echo $i
    python mat_copy.py --size $i
    # python mat_scale.py --size $i
    # python mat_axpy.py --size $i
    # python mat_xpxpy.py --size $i
    # or do whatever with individual element of the array
done