#!/bin/bash

# sizes=(4096  8192 16384  65536  262144 524288 1048576  4194304  16777216  67108864  268435456 536870912)
sizes=(4096  8192 16384  65536  262144 524288 1048576  4194304  16777216  67108864)
## now loop through the above array
for i in ${sizes[@]}
do
#    echo $i
    # bin/mat_copy -size=$i #-block0=512 -block1=1
    # bin/mat_scale -size=$i
    # bin/mat_axpy -size=$i
    bin/mat_xpxpy -size=$i #-blockx=256 -blocky=1
   # or do whatever with individual element of the array
done