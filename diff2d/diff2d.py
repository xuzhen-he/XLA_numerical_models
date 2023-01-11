import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import bench

import jax.numpy as jnp


class diff2d(bench.bench2d):

    def __init__(self):
        super().__init__()

        self.delta_x = 2.0 / (self.side_size - 1.0)
        self.a = 1.0
        self.r = 0.2
        self.delta_t = self.r * self.delta_x * self.delta_x / self.a
        self.t0 = 0.001

        print("\nSimulation info: 2D diffusion equation")
        print("  Domain from -1 to 1")
        print("  dx = ", self.delta_x)
        print("  a = ", self.a)
        print("  r = a*dt/dx: ", self.r)
        print("  dt = ", self.delta_t)
        print("  t0 = ", self.t0)
        print("  u at centre initially: ", self.analytical(0.0, 0.0, self.t0))
     
    def analytical(self, x, y, t):
        return jnp.exp(-(x * x + y * y) /
                       (4.0 * self.a * t)) / (4.0 * jnp.pi * self.a * t)

    def analytical_mesh(self, dtype, t):
        ll = 1.0
        xx = jnp.linspace(-ll, ll, self.side_size, dtype=dtype)
        yy = jnp.linspace(-ll, ll, self.side_size, dtype=dtype)
        [x, y] = jnp.meshgrid(xx, yy, indexing='ij')
        return self.analytical(x, y, t)

    def initial_condition(self, dtype):
        return self.analytical_mesh(dtype, self.t0)

    def test_result(self, u, t):

        ua = self.analytical_mesh(u.dtype, t)

        mid = int(self.side_size / 2)
        # plt_step = max(math.floor(self.side_size / 500), 1)
        # plt.plot(u[mid, ::plt_step])
        # plt.plot(ua[mid, ::plt_step])
        # plt.show()

        error = ua - u

        print(" \nCheck result")
        print("  u at centre: ", u[mid, mid])
        print("  u analytical at centre: ", ua[mid, mid])
        print("  mean squared error: ", jnp.linalg.norm(error))
        print("  mean squared analytical: ", jnp.linalg.norm(ua))
