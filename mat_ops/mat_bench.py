import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import bench

import jax.numpy as jnp


class mat_bench(bench.bench2d):

    def __init__(self):
        super().__init__()
     
    def test_result(self, y, target):
        sum = jnp.sum(y)
        if target == 0.0:
            pass_test = abs(sum - target) < 1.0e-10
        else:
            pass_test = abs(sum - target) / target < 1.0e-10
        print(" \nCheck result")
        print("  sum y ", sum)
        print("  equal" if pass_test else "  not equal")
        print("  target ", target)
