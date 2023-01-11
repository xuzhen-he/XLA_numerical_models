import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

data = np.loadtxt("residual")
c = 10
    
ax.semilogy(data[c:, 0], data[c:, 1], label="u", linewidth=0.6)
ax.semilogy(data[c:, 0], data[c:, 2], label="v", linewidth=0.6)
ax.semilogy(data[c:, 0], data[c:, 3], label="p", linewidth=0.6)


ax.tick_params(direction='out', top=False, right=False)
ax.set_title("Residual")
ax.set_xlabel('iteration')
ax.set_ylabel('Residual')
plt.tight_layout()
ax.legend(loc="best")
plt.show()