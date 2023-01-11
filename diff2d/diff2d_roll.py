import matplotlib.pyplot as plt
import time

import diff2d

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
dtype = jnp.dtype(jnp.float64)
# jax.config.update('jax_platform_name', 'cpu')


@jax.jit
def kernel(u, r):
    print("Tracing Jax function ...............")
    u = (1 - 4.0 * r) * u + r * (jnp.roll(u, [-1], axis=0) + jnp.roll(
        u, [1], axis=0) + jnp.roll(u, [-1], axis=1) + jnp.roll(u, [1], axis=1))

    u = u.at[0, :].set(0.0)  #x = 0 y = all
    u = u.at[-1, :].set(0.0)  #x = end, y = all
    u = u.at[1:-1, 0].set(0.0)  #x = all except for two ends y = 0
    u = u.at[1:-1, -1].set(0.0)  #x = all except for two ends, y = end

    return u


class diff2d_stack(diff2d.diff2d):

    def __init__(self):
        super().__init__()
        self.elem_size = dtype.itemsize
        self.memory_transfer_per_loop = 2.0 * float(self.elem_size) * float(
            self.total_size) / (1024.0 * 1024.0 * 1024.0)

    def benchmark(self):
        self.print_bench()

        u = self.initial_condition(dtype)
        self.assert_shape_and_dtype(u)
        # print(u.shape)
        rr = jnp.array([self.r], dtype=dtype)

        # u = jnp.reshape(jnp.arange(100),(10, 10))
        # print(u)
        # test = jnp.roll(u, [-1], axis=0)
        # print(test)

        # warm up
        u = kernel(u, rr)

        self.loops = 0
        start_time = time.perf_counter()
        while ((time.perf_counter() - start_time) < self.benchtime):
            # while(self.loops<100):
            u = kernel(u, rr)
            self.loops = self.loops + 1
        u.block_until_ready()
        self.duration = time.perf_counter() - start_time

        t = self.delta_t * self.loops + self.t0
        self.test_result(u, t)
        self.print_performance()


if __name__ == "__main__":
    test = diff2d_stack()
    test.benchmark()