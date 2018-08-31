import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.stats import multivariate_normal



# positions in 2 axis of points to plot on
x, y = np.mgrid[-1.0:1.0:0.1, -1.0:1.0:0.1]

# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat])

# mean
mu = np.array([0.0, 0.0])
# stddev
sigma = np.array([0.5, 0.5])
# covariance matrix
covariance = np.diag(sigma**2)

# points sampled from the normal distribution
z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

# Reshape back to a (30, 30) grid.
z = z.reshape(x.shape)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(x,y,z)
#ax.plot_wireframe(x,y,z)

plt.show()
