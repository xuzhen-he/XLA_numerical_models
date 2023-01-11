import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import bench

import jax.numpy as jnp


class diff1d(bench.bench):

    def __init__(self):
        super().__init__()

        self.delta_x = 1.0 / (self.total_size - 1.0)
        self.a = 1.0
        self.r = 0.4
        self.delta_t = self.r * self.delta_x * self.delta_x / self.a

        print("\nSimulation info: 1D diffusion equation")
        print("  x from 0 to 1")
        print("  a = ", self.a)
        print("  r = a*Dt/Dx: ", self.r)

    def initial_condition(self, dtype):
        x = jnp.linspace(0.0, 1.0, self.total_size, dtype=dtype)
        return 6.0 * jnp.sin(math.pi * x)

    def test_result(self, u, t):

        decay = math.exp(-self.a * math.pi * math.pi * t)
        mid = int(self.total_size / 2)
        midx = float(mid) * self.delta_x
        umidf_analytical = 6.0 * math.sin(math.pi * midx) * decay
        umidf = u[mid]

        x = jnp.linspace(0.0, 1.0, self.total_size, dtype=u.dtype)
        u_analytical = 6.0 * jnp.sin(math.pi * x) * decay
        sum_ua2 = jnp.linalg.norm(u_analytical)
        # plt.plot(x[::plt_step], u[::plt_step])
        # plt.scatter(x[::plt_step], u_analytical[::plt_step])
        # plt.show()

        error = u - u_analytical
        sum_e2 = jnp.linalg.norm(error)

        print("Check result")
        print("  u at centre analyticallly: ", umidf_analytical)
        print("  u at centre numerically: ", umidf)
        print("  sum of u analyticall: ", sum_ua2)
        print("  sum of error: ", sum_e2)
