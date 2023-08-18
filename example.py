from rdp_quick import rdp_single_initial_window, rdp_points_per_window
import numpy as np
import time
import matplotlib.pyplot as plt

num_osc = 20
points_per_osc = 200
epsilon = 0.01

print("building points")
n_pts = num_osc*points_per_osc
x = np.arange(n_pts)/n_pts*num_osc*2.0*np.pi
y = np.sin(x)
p = np.vstack([x, y]).astype(float).transpose()

print("running")
st = time.time()
down_sampled_p = rdp_single_initial_window(p, epsilon)
et = time.time()
print(et - st)
print(down_sampled_p.shape, p.shape)

fig, ax = plt.subplots(1, 1)
ax.plot(x, y, "-s", label="initial points")
ax.plot(down_sampled_p[:, 0], down_sampled_p[:, 1], "-*", label="selected points")
ax.legend()

print("running points per window")
st = time.time()
down_sampled_p = rdp_points_per_window(p, epsilon, points_per_osc)
et = time.time()
print(et - st)
print(down_sampled_p.shape, p.shape)

fig, ax = plt.subplots(1, 1)
ax.plot(x, y, "-s", label="initial points")
ax.plot(down_sampled_p[:, 0], down_sampled_p[:, 1], "-*", label="selected points")
ax.legend()

plt.show()
