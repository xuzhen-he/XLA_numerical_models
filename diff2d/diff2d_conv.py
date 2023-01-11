import matplotlib.pyplot as plt
import time

import diff2d

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
dtype = jnp.dtype(jnp.float64)
# jax.config.update('jax_platform_name', 'cpu')


@jax.jit
def kernel(u, filter, zeros_h, zeros_v):
    print("Tracing Jax function ...............")
    inner = jax.scipy.signal.convolve(u, filter, mode='valid')
    inner_with_top_bot = jnp.concatenate([zeros_h, inner, zeros_h], axis=0)
    u = jnp.concatenate([zeros_v, inner_with_top_bot, zeros_v], axis=1)
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

        filter = jnp.array([[0.0, self.r, 0.0],
                            [self.r, (1 - 4.0 * self.r), self.r],
                            [0.0, self.r, 0.0]]).astype(dtype)
        # print(filter)
        zeros_h = jnp.zeros((1, self.side_size - 2), dtype=dtype)
        zeros_v = jnp.zeros((self.side_size, 1), dtype=dtype)

        # inner = jax.scipy.signal.convolve(u, filter,mode='valid')
        # print(inner.shape)
        # inner_with_top_bot = jnp.concatenate([zeros_h, inner, zeros_h], axis=0)
        # print(inner_with_top_bot.shape)
        # u = jnp.concatenate([zeros_v, inner_with_top_bot, zeros_v], axis=1)
        # print(u.shape)

        # warm up
        u = kernel(u, filter, zeros_h, zeros_v)

        self.loops = 0
        start_time = time.perf_counter()
        while ((time.perf_counter() - start_time) < self.benchtime):
            # while(self.loops<100):
            u = kernel(u, filter, zeros_h, zeros_v)
            self.loops = self.loops + 1
        u.block_until_ready()
        self.duration = time.perf_counter() - start_time

        t = self.delta_t * self.loops + self.t0
        self.test_result(u, t)
        self.print_performance()


if __name__ == "__main__":
    test = diff2d_stack()
    test.benchmark()