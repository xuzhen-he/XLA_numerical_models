import os, sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import mat_bench

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
dtype = jnp.dtype(jnp.float32)
# jax.config.update("jax_platform_name", "cpu")


@jax.jit
def kernel(a, x, y):
    print("Tracing Jax function ...............")
    return a * x + y


class mat_axpy(mat_bench.mat_bench):

    def __init__(self):
        super().__init__()
        self.elem_size = dtype.itemsize
        self.memory_transfer_per_loop = (3.0 * float(self.elem_size) *
                                         float(self.total_size) /
                                         (1024.0 * 1024.0 * 1024.0))

    def benchmark(self):
        self.print_bench()
        print("\nSimulation info: 2d mat axpy\n")

        x = jnp.ones((self.side_size, self.side_size), dtype=dtype)
        y = jnp.zeros((self.side_size, self.side_size), dtype=dtype)
        a = 1.0
        self.assert_shape_and_dtype(x)
        self.assert_shape_and_dtype(y)

        # warm up
        y = kernel(a, x, y)

        #main loop
        self.loops = 0
        start_time = time.perf_counter()
        while ((time.perf_counter() - start_time) < self.benchtime):
            y = kernel(a, x, y)
            self.loops = self.loops + 1
        y.block_until_ready()
        self.duration = time.perf_counter() - start_time

        target = float(self.total_size * (self.loops + 1))
        self.test_result(y, target)
        self.print_performance()


if __name__ == "__main__":
    test = mat_axpy()
    test.benchmark()
