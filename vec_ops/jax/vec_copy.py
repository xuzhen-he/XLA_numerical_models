import os, sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import vec_bench

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
dtype = jnp.dtype(jnp.float64)
# jax.config.update("jax_platform_name", "cpu")


@jax.jit
def kernel(x):
    print("\nTracing Jax function ...............")
    return jnp.copy(x)


class vec_copy(vec_bench.vec_bench):

    def __init__(self):
        super().__init__()
        self.elem_size = dtype.itemsize
        self.memory_transfer_per_loop = (2.0 * float(self.elem_size) *
                                         float(self.total_size) /
                                         (1024.0 * 1024.0 * 1024.0))

    def benchmark(self):
        self.print_bench()
        print("\nSimulation info: 1d vec copy\n")

        x = jnp.ones((self.total_size, ), dtype=dtype)
        y = jnp.zeros((self.total_size, ), dtype=dtype)
        self.assert_shape_and_dtype(x)
        self.assert_shape_and_dtype(y)

        # warm up
        y = kernel(x)

        #main loop
        self.loops = 0
        start_time = time.perf_counter()
        while ((time.perf_counter() - start_time) < self.benchtime):
            y = kernel(x)
            self.loops = self.loops + 1
        y.block_until_ready()
        self.duration = time.perf_counter() - start_time

        self.test_result(y, float(self.total_size))
        self.print_performance()


if __name__ == "__main__":
    test = vec_copy()
    test.benchmark()
