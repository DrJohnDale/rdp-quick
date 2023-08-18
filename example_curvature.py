from rdp_quick import rdp_windows_from_curvature
import numpy as np
import time
import matplotlib.pyplot as plt

num_osc = 20
points_per_osc = 200
epsilon = 0.01

print("building points")
n_pts = num_osc*points_per_osc
x = np.arange(n_pts)/n_pts*num_osc*2.0*np.pi
y = np.sin(x)*2
p = np.vstack([x, y]).astype(float).transpose()

print("running")
st = time.time()
down_sampled_p = rdp_windows_from_curvature(p, epsilon)
et = time.time()
print(et - st)
print(down_sampled_p.shape, p.shape)

fig, ax_data = plt.subplots(1, 1, sharex=True)
ax_data.plot(x, y, "-*", label="initial points")
ax_data.plot(down_sampled_p[:, 0], down_sampled_p[:, 1], "-*", label="selected points")
ax_data.legend()
plt.show()
