import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from matplotlib import cm
# from mpl_toolkits.axes_grid1 import make_axes_locatable

exp_label = 'Re100'

u = np.loadtxt('u.txt')
v = np.loadtxt('v.txt')
p = np.loadtxt('p.txt')

# u = np.loadtxt('test_u',delimiter=',')
# v = np.loadtxt('test_v',delimiter=',')
# p = np.loadtxt('test_p',delimiter=',')

side_size = u.shape[0]

ll = 1.0
xx = np.linspace(0.0, ll, side_size)
yy = np.linspace(0.0, ll, side_size)
[x, y] = np.meshgrid(xx, yy, indexing='ij')

rc = 1
if side_size % 2 == 0:
    um = 0.5 * (u[int(side_size / 2), :] + u[int(side_size / 2) + 1, :])
    vm = 0.5 * (v[:, int(side_size / 2)] + v[:, int(u.shape[0] / 2) + 1])
else:
    um = u[int((u.shape[0] + 1)/ 2), :]
    vm = v[:, int((u.shape[0] + 1)/ 2)]
# vm = v[int(v.shape[0] / 2), :]

exp_uy = pd.read_csv('Ghia_uy.csv')
exp_vx = pd.read_csv('Ghia_vx.csv')
# print(exp_uy)
# print(exp_uy[exp_label])

fig = plt.figure()
ax_u = fig.add_subplot(111)

ax_v = ax_u.twinx()
ax_v2 = ax_v.twiny()

ax_u.plot(um, yy, color='g', label="Numerical, $U_x$", linewidth=0.6)
ax_v2.plot(xx, vm, label="Numerical, $U_y$", linewidth=0.6)

ax_u.plot(
    exp_uy[exp_label],
    exp_uy['y'],
    '*',
    color='r',
    label="Ghia, $U_x$",
    linewidth=0.6)


ax_v2.plot(
    exp_vx['x'],
    exp_vx[exp_label],
    'x',
    label="Ghia, $U_y$",
    linewidth=0.6)

ax_u.set_xlim([-1, 1])
ax_u.set_ylim([0, 1])
ax_v.set_ylim([-1, 1])
ax_v2.set_xlim([0, 1])

ax_u.set_title("Velocity profile along the middle of axes")

# ax_u.set_ylabel('$y$ (m)')
# ax_v2.set_xlabel('$x$ (m)')
# ax_u.set_xlabel('$U_y$ (m.s$^{-1}$)')
# ax_v.set_ylabel('$U_x$ (m.s$^{-1}$)')

ax_u.set_aspect('auto')
ax_v.set_aspect('auto')

plt.tight_layout()
plt.show()
