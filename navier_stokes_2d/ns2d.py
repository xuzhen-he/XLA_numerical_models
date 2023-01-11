import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import bench

import jax.numpy as jnp


class ns2d(bench.bench2d):

    def __init__(self):
        super().__init__()
        self.parser.add_argument('--re',
                            type=float,
                            action='store',
                            default=100.0,
                            help='Reynolds')
        self.parser.add_argument('--ma',
                            type=float,
                            action='store',
                            default=0.1,
                            help='Mach')

        args = self.parser.parse_args()
        re = args.re
        ma = args.ma

        self.u0 = 1.0
        ll = 1.0
        nu = self.u0 * ll / re
        sound_speed = self.u0 / ma

        dx = ll / (self.side_size - 1.0)
        dy = ll / (self.side_size - 1.0)

        safe_factor_CFL = 0.2
        safe_factor_diffusion = 0.1
        dt_CFL = safe_factor_CFL * ma * dx
        dt_Re = safe_factor_diffusion * re * dx * dx
        dt = min(dt_CFL, dt_Re)

        self.dtdx = dt / dx
        self.dtdy = dt / dy
        dtdxx = dt / (dx * dx)
        dtdyy = dt / (dy * dy)
        self.nu_dtdxx = nu * dtdxx
        self.nu_dtdyy = nu * dtdyy
        self.c2_dtdx = sound_speed * sound_speed * self.dtdx
        self.c2_dtdy = sound_speed * sound_speed * self.dtdy

        print("\nSimulation info: 2D cavity flow")
        print("  Domain x from 0 to 1, y from 0 to 1")
        print("  Re: ", re)
        print("  Ma: ", ma)
        print("  dx = ", dx)
        print("  dt = ", dt)
        print("  lid velocity = ", self.u0)

    def initial_condition(self, dtype):
        u = jnp.zeros((self.side_size, self.side_size), dtype=dtype)
        v = jnp.zeros((self.side_size, self.side_size), dtype=dtype)
        p = jnp.zeros((self.side_size, self.side_size), dtype=dtype)
        u = u.at[1:self.side_size - 1, self.side_size - 1].set(self.u0)
        return u, v, p

    def test_result(self, u, t):
        pass
        # ua = self.analytical_mesh(u.dtype, t)

        # mid = int(self.side_size / 2)
        # # plt_step = max(math.floor(self.side_size / 500), 1)
        # # plt.plot(u[mid, ::plt_step])
        # # plt.plot(ua[mid, ::plt_step])
        # # plt.show()

        # error = ua - u

        # print(" \nCheck result")
        # print("  u at centre: ", u[mid, mid])
        # print("  u analytical at centre: ", ua[mid, mid])
        # print("  mean squared error: ", jnp.linalg.norm(error))
        # print("  mean squared analytical: ", jnp.linalg.norm(ua))
