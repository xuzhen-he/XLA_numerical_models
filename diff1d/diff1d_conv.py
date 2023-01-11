import sys, os
import time
import math

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import diff1d

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
dtype = jnp.dtype(jnp.float32)
# jax.config.update('jax_platform_name', 'cpu')

##numpy.convolve
# @jax.jit
# def kernel(u, filter, zero):
#     print("Tracing Jax function ...............")
#     return jnp.concatenate([zero, jnp.convolve(u,filter,mode='valid'), zero], axis=0)

##jax.scipy.signal.convolve
# @jax.jit
# def kernel(u, filter, zero):
#     print("Tracing Jax function ...............")
#     return jnp.concatenate([zero, jax.scipy.signal.convolve(u, filter,mode='valid'), zero], axis=0)


##jax.lax.conv
@jax.jit
def kernel(u, filter, zero):
    print("Tracing Jax function ...............")
    return jnp.concatenate([zero, jax.lax.conv(u,filter,(1,),'VALID'), zero], axis=2)

class diff1d_conv(diff1d.diff1d):

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
        filter = jnp.stack([self.r, (1- 2.0 * self.r), self.r]).squeeze().astype(dtype)
        # print(filter.shape)

        #####jnp.convolve######
        # test = jnp.convolve(u,filter,mode='valid')
        # # print(test.shape)
        # test1 = jnp.concatenate([zero, test, zero], axis=0)
        # print(test1.shape)
        # u = jnp.expand_dims(u, axis=(0, 1))
        # filter = jnp.expand_dims(filter, axis=(0, 1))

        #####jax.scipy.signal.convolve######
        # test = jax.scipy.signal.convolve(u, filter,mode='valid')
        # print(test.shape)

        #####lax.conv#####
        u = jnp.expand_dims(u, axis=(0, 1))
        # print(u.shape)
        # print(u.squeeze().shape)
        filter = jnp.expand_dims(filter, axis=(0, 1))
        zero = jnp.expand_dims(zero, axis=(0, 1))
        # test = jax.lax.conv(u,filter,(1,),'VALID')
        # print(test.shape)
        # test1 = jnp.concatenate([zero, test, zero], axis=2)
        # print(test1.shape)

        # plt_step = math.floor(self.node_size / 500.0)
        # plt.plot(x[::plt_step], u[::plt_step])

        #warm up
        u = kernel(u, filter, zero)

        #main loop
        self.loops = 0
        start_time = time.perf_counter()
        while ((time.perf_counter() - start_time) < self.benchtime):
            u = kernel(u, filter, zero)
            self.loops = self.loops + 1
        u.block_until_ready()
        self.duration = time.perf_counter() - start_time

        t = self.delta_t * (self.loops + 1)
        self.test_result(u.squeeze(), t)
        self.print_performance()


if __name__ == "__main__":
    test = diff1d_conv()
    test.benchmark()
