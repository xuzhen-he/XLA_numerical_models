import sys, os
import time
import math

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import diff1d

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
dtype = jnp.dtype(jnp.float64)
# jax.config.update('jax_platform_name', 'cpu')


@jax.jit
def kernel(u, r):
    print("Tracing Jax function ...............")
    unew = (1 - 2.0 * r) * u + r * (jnp.roll(u, [-1], axis=0) + jnp.roll(u, [1], axis=0))
    unew = unew.at[0].set(0.0)
    unew = unew.at[-1].set(0.0)
    return unew

class diff1d_stack(diff1d.diff1d):

    def __init__(self):
        super().__init__()
        self.elem_size = dtype.itemsize
        self.memory_transfer_per_loop = (2.0 * float(self.elem_size) *
                                         float(self.total_size) /
                                         (1024.0 * 1024.0 * 1024.0))

    def benchmark(self):
        self.print_bench()

        u = self.initial_condition(dtype)
        self.assert_shape_and_dtype(u)

        zero = jnp.array([0.0], dtype=dtype)
        r = jnp.array([self.r], dtype=dtype)

        # plt_step = math.floor(self.node_size / 500.0)
        # plt.plot(x[::plt_step], u[::plt_step])

        #warm up
        u = kernel(u, r)

        #main loop
        self.loops = 0
        start_time = time.perf_counter()
        while ((time.perf_counter() - start_time) < self.benchtime):
            u = kernel(u, r)
            self.loops = self.loops + 1
        u.block_until_ready()
        self.duration = time.perf_counter() - start_time

        t = self.delta_t * (self.loops + 1)
        self.test_result(u, t)
        self.print_performance()


if __name__ == "__main__":
    test = diff1d_stack()
    test.benchmark()
