#!/bin/bash

# sizes=(64  256  1024  4096  16384  65536  262144  1048576  4194304  16777216  67108864  268435456 536870912 1073741824)
sizes=(4096  8192 16384  65536  262144 524288 1048576  4194304  16777216  67108864  268435456 536870912)

## now loop through the above array
for i in ${sizes[@]}
do
#    echo $i
    # bin/mat_copy -size=$i
    # bin/mat_scale -size=$i
    # bin/mat_axpy -size=$i
    bin/mat_xpxpy -size=$i
   # or do whatever with individual element of the array
done