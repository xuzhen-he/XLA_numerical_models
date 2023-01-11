#!/bin/bash

sizes=(64  256  1024  4096  16384  65536  262144  1048576  4194304  16777216  67108864  268435456)
## now loop through the above array
for i in ${sizes[@]}
do
#    echo $i
    # bin/vec_copy -size=$i
    # bin/vec_scale -size=$i
    # bin/vec_axpy -size=$i
    bin/vec_xpxpy -size=$i
   # or do whatever with individual element of the array
done