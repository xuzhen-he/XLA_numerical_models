import matplotlib.pyplot as plt
import time

import diff2d

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
dtype = jnp.dtype(jnp.float64)
# jax.config.update('jax_platform_name', 'cpu')


@jax.jit
def kernel(u, r, zeros_h, zeros_v):
    print("Tracing Jax function ...............")
    inner = (1 - 4.0 * r) * u[1:-1, 1:-1] + r * (u[:-2, 1:-1] + u[2:, 1:-1] +
                                                 u[1:-1, :-2] + u[1:-1, 2:])
    inner_with_top_bot = jnp.concatenate([zeros_h, inner, zeros_h], axis=0)
    u = jnp.concatenate([zeros_v, inner_with_top_bot, zeros_v], axis=1)
    return u

# @jax.jit
# def update(u, r):
#     print("Tracing Jax function ...............")
#     u = u.at[1:-1, 1:-1].set((1 - 4.0 * r) * u[1:-1, 1:-1]  #u_i_j
#                              + r * (
#                                  u[:-2, 1:-1]  #u_i-1_j
#                                  + u[2:, 1:-1]  #u_i+1_j
#                                  + u[1:-1, :-2]  #u_i_j-1
#                                  + u[1:-1, 2:]))  #u_i_j+1

#     # good implementation
#     u = u.at[0, :].set(0.0)  #x = 0 y = all
#     u = u.at[-1, :].set(0.0)  #x = end, y = all
#     u = u.at[1:-1, 0].set(0.0)  #x = all except for two ends y = 0
#     u = u.at[1:-1, -1].set(0.0)  #x = all except for two ends, y = end

#     # bad implementation
#     # u = u.at[:, 0].set(0.0)  # y = 0 x = all
#     # u = u.at[:, -1].set(0.0)  # y = end, x = all
#     # u = u.at[0, 1:-1].set(0.0)  #y = all except for two ends x = 0
#     # u = u.at[-1, 1:-1].set(0.0)  #y = all except for two ends, x = end
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
        # print(u)
        zeros_h = jnp.zeros((1, self.side_size - 2), dtype=dtype)
        zeros_v = jnp.zeros((self.side_size, 1), dtype=dtype)
        rr = jnp.array([self.r], dtype=dtype)
        # mid = int(self.side_size / 2)
        # plt_step = max(math.floor(self.side_size / 500),1)
        # plt.plot(u[mid, ::plt_step])
        # plt.show()

        # warm up
        u = kernel(u, rr, zeros_h, zeros_v)

        self.loops = 0
        start_time = time.perf_counter()
        while ((time.perf_counter() - start_time) < self.benchtime):
            # while(self.loops<100):
            u = kernel(u, rr, zeros_h, zeros_v)
            self.loops = self.loops + 1
        u.block_until_ready()
        self.duration = time.perf_counter() - start_time

        t = self.delta_t * self.loops + self.t0
        self.test_result(u, t)
        self.print_performance()


if __name__ == "__main__":
    test = diff2d_stack()
    test.benchmark()