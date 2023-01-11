# import os
import time

import jax
import jax.numpy as jnp
# import numpy as np
# import matplotlib.pyplot as plt

import ns2d

jax.config.update("jax_enable_x64", True)
dtype = jnp.dtype(jnp.float64)
# jax.config.update("jax_platform_name", "cpu")


@jax.jit
def predictor(u, v, p, \
            dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
            zeros0, zeros1, u0_array):

    print("Tracing Jax function predictor...............")

    ustar_inner = u[1:-1, 1:-1] \
        - dtdx * (u[1:-1, 1:-1] *(u[2:, 1:-1] - u[1:-1, 1:-1]) + p[2:, 1:-1] - p[1:-1, 1:-1]) \
        - dtdy * (v[1:-1, 1:-1] * (u[1:-1, 2:] - u[1:-1, 1:-1])) \
        + nu_dtdxx * (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) \
        + nu_dtdyy * (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2])
    # print(ustar_inner.shape)
    # print(zeros0.shape)
    # print(u0_array.shape)
    ustar = jnp.concatenate(
        [zeros1, \
        jnp.concatenate([zeros0, ustar_inner, u0_array], axis=1), \
        zeros1],axis=0)
    # print(ustar.shape)
    # print(ustar)

    vstar_inner = v[1:-1, 1:-1] \
        - dtdx * (u[1:-1, 1:-1] *(v[2:, 1:-1] - v[1:-1, 1:-1])) \
        - dtdy * (v[1:-1, 1:-1] *(v[1:-1, 2:] - v[1:-1, 1:-1]) + p[1:-1, 2:] - p[1:-1, 1:-1]) \
        + nu_dtdxx * (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1]) \
        + nu_dtdyy * (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, :-2])
    vstar = jnp.concatenate(
        [zeros1, \
        jnp.concatenate([zeros0, vstar_inner, zeros0], axis=1), \
        zeros1],axis=0)
    # print(vstar.shape)
    # print(vstar)

    pstar_inner = p[1:-1, 1:-1] \
                -c2_dtdx * (u[2:, 1:-1] - u[1:-1, 1:-1]) \
                -c2_dtdy * (v[1:-1, 2:] - v[1:-1, 1:-1])
    # print(pstar_inner.shape)
    # print(jnp.expand_dims(pstar_inner[:, 0], axis=1).shape)
    pstar_p1 = jnp.concatenate(
        [jnp.expand_dims(pstar_inner[:, 0], axis=1),\
        pstar_inner,\
        jnp.expand_dims(pstar_inner[:, -1], axis=1)], axis=1)
    # print(pstar_p1.shape)

    pstar = jnp.concatenate(
        [jnp.expand_dims(pstar_p1[0, :], axis=0),\
            pstar_p1,\
            jnp.expand_dims(pstar_p1[-1, :], axis=0)], axis=0)
    # print(pstar.shape)
    return ustar, vstar, pstar


@jax.jit
def corrector(u,v,p,\
            ustar,vstar,pstar,\
            dtdx,dtdy,nu_dtdxx,nu_dtdyy,c2_dtdx,c2_dtdy,\
            zeros0,zeros1,u0_array):

    print("Tracing Jax function corrector...............")
    # print(ustar.shape)
    # print(vstar.shape)
    # print(pstar.shape)
    ustar2_inner = ustar[1:-1, 1:-1] \
        - dtdx * (ustar[1:-1, 1:-1] *(ustar[1:-1, 1:-1] - ustar[:-2, 1:-1]) +pstar[1:-1, 1:-1] - pstar[:-2, 1:-1]) \
        - dtdy * (vstar[1:-1, 1:-1] *(ustar[1:-1, 1:-1] - ustar[1:-1, :-2])) \
        + nu_dtdxx *(ustar[2:, 1:-1] - 2.0 * ustar[1:-1, 1:-1] + ustar[:-2, 1:-1]) \
        + nu_dtdyy *(ustar[1:-1, 2:] - 2.0 * ustar[1:-1, 1:-1] + ustar[1:-1, :-2])
    # print(ustar2_inner.shape)
    ustar2 = jnp.concatenate(
        [zeros1, \
        jnp.concatenate([zeros0, ustar2_inner, u0_array], axis=1), \
        zeros1],axis=0)
    # print(ustar2.shape)

    vstar2_inner = vstar[1:-1, 1:-1] \
        - dtdx * (ustar[1:-1, 1:-1] *(vstar[1:-1, 1:-1] - vstar[:-2, 1:-1])) \
        - dtdy * (vstar[1:-1, 1:-1] * (vstar[1:-1, 1:-1] - vstar[1:-1, :-2]) + pstar[1:-1, 1:-1] - pstar[1:-1, :-2]) \
        + nu_dtdxx * (vstar[2:, 1:-1] - 2.0 * vstar[1:-1, 1:-1] + vstar[:-2, 1:-1]) \
        + nu_dtdyy * (vstar[1:-1, 2:] - 2.0 * vstar[1:-1, 1:-1] + vstar[1:-1, :-2])
    vstar2 = jnp.concatenate(
        [zeros1, \
        jnp.concatenate([zeros0, vstar2_inner, zeros0], axis=1), \
        zeros1],axis=0)
    # print(vstar2.shape)

    pstar2_inner = pstar[1:-1, 1:-1] \
            - c2_dtdx * (ustar[1:-1, 1:-1] - ustar[:-2, 1:-1]) \
            - c2_dtdy * (vstar[1:-1, 1:-1] - vstar[1:-1, :-2])
    pstar2_p1 = jnp.concatenate(
        [jnp.expand_dims(pstar2_inner[:, 0], axis=1),\
        pstar2_inner,\
        jnp.expand_dims(pstar2_inner[:, -1], axis=1)], axis=1)
    pstar2 = jnp.concatenate(
        [jnp.expand_dims(pstar2_p1[0, :], axis=0),\
            pstar2_p1,\
            jnp.expand_dims(pstar2_p1[-1, :], axis=0)], axis=0)
    # print(pstar2.shape)

    u = 0.5 * (u + ustar2)
    v = 0.5 * (v + vstar2)
    p = 0.5 * (p + pstar2)
    return u, v, p


class ns2d_stack(ns2d.ns2d):

    def __init__(self):
        super().__init__()
        self.elem_size = dtype.itemsize
        self.memory_transfer_per_loop = (15.0 * float(self.elem_size) *
                                         float(self.total_size) /
                                         (1024.0 * 1024.0 * 1024.0))

    def benchmark(self):
        self.print_bench()

        u, v, p = self.initial_condition(dtype)
        self.assert_shape_and_dtype(u)
        self.assert_shape_and_dtype(v)
        self.assert_shape_and_dtype(p)
        # print(u)

        dtdx = jnp.array([self.dtdx], dtype=dtype)
        dtdy = jnp.array([self.dtdy], dtype=dtype)
        nu_dtdxx = jnp.array([self.nu_dtdxx], dtype=dtype)
        nu_dtdyy = jnp.array([self.nu_dtdyy], dtype=dtype)
        c2_dtdx = jnp.array([self.c2_dtdx], dtype=dtype)
        c2_dtdy = jnp.array([self.c2_dtdy], dtype=dtype)

        zeros0 = jnp.zeros((self.side_size - 2, 1), dtype=dtype)
        zeros1 = jnp.zeros((1, self.side_size), dtype=dtype)
        u0_array = self.u0 * jnp.ones((self.side_size - 2, 1), dtype=dtype)

        # warm up
        ustar, vstar, pstar = predictor(u, v, p, \
                            dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
                            zeros0, zeros1, u0_array)
        # print(ustar)
        u, v, p= corrector(u,v,p,\
                ustar,vstar,pstar,\
                dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
                zeros0, zeros1, u0_array)
        # print(u)

        # log_name = 'residual'
        # if os.path.exists(log_name):
        #     os.remove(log_name)
        # flog = open(log_name, 'ab')

        e0 = self.get_u_increment(u,v,p,\
                    dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
                    zeros0, zeros1, u0_array)

        self.loops = 0
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < self.benchtime:
            # while (self.loops < 200000):
            ustar, vstar, pstar = predictor(u, v, p, \
                                dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
                                zeros0, zeros1, u0_array)

            u, v, p= corrector(u,v,p,\
                    ustar,vstar,pstar,\
                    dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
                    zeros0, zeros1, u0_array)

            # np.savetxt(flog,
            #            np.c_[self.loops,
            #                  jnp.linalg.norm(ua),
            #                  jnp.linalg.norm(va),
            #                  jnp.linalg.norm(pa)],
            #            fmt='%.8f')

            self.loops = self.loops + 1

        p.block_until_ready()
        self.duration = time.perf_counter() - start_time

        ef = self.get_u_increment(u,v,p,\
                    dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
                    zeros0, zeros1, u0_array)
        print(" \nCheck result")
        print("  u incremet initiallly: ", e0)
        print("  u incremet last step: ", ef)
        print("  ratio: ", ef / e0)

        self.print_performance()

        # np.savetxt('u.txt', u)
        # np.savetxt('v.txt', v)
        # np.savetxt('p.txt', p)

    def get_u_increment(self, u, v, p,\
                    dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
                    zeros0, zeros1, u0_array):

        u_inc = -u
        ustar, vstar, pstar = predictor(u, v, p, \
                                dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
                                zeros0, zeros1, u0_array)

        u, v, p= corrector(u,v,p,\
                    ustar,vstar,pstar,\
                    dtdx, dtdy, nu_dtdxx, nu_dtdyy, c2_dtdx, c2_dtdy,\
                    zeros0, zeros1, u0_array)
        u_inc = u_inc + u
        return jnp.linalg.norm(u_inc)


if __name__ == "__main__":
    test = ns2d_stack()
    test.benchmark()
