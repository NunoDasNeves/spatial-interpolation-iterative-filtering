
import numpy as np
from scipy.stats import multivariate_normal
from numpy.random import rand, normal, random_integers, uniform

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim


_CUBE_SIZE=10


def plot_func(T, func):
    '''
        Assuming 1 dimension, plot func
    '''
    # positions in 2 axis of points to plot on
    # TODO right now this -1 +1 business is a hack to get contour() to plot everything
    x, y = np.mgrid[-1:T+1:1, -_CUBE_SIZE:CUBE_SIZE:0.1]

    z = np.array([func(t,T,[i]) for t,i in zip(np.ravel(x), np.ravel(y))])
    # Reshape back to a grid.
    z = z.reshape(x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax.plot_surface(x,y,z)
    #ax.plot_wireframe(x,y,z)
    ax.contour(x,y,z, zdir='x', colors=['orange','red','purple','blue','green'], levels=range(T))

    plt.show()

def plot_observations(T, func, sensor_pos, sensor_data):
    '''
        Assuming 1 dimension, plot func vs sensor data
    '''
    # positions in 2 axis of points to plot on
    x, y = np.mgrid[-1:T+1:1, -10.0:10.0:0.1]

    z = np.array([func(t,T,[i]) for t,i in zip(np.ravel(x), np.ravel(y))])
    # Reshape back to a grid.
    z = z.reshape(x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.contour(x,y,z, zdir='x', colors=['orange','red','purple','blue','green'], levels=range(T))

    # position in time
    sensor_x = []
    # position in space
    sensor_y = []
    # actual value
    sensor_z = []
    # we need T*N data points
    for i in range(T):
        for j in range(len(sensor_pos[0])):
            sensor_x.append(i)
            sensor_y.append(sensor_pos[i][j])
            sensor_z.append(sensor_data[i][j])

    ax.scatter(sensor_x, sensor_y, sensor_z)

    plt.show()



def moving_gaussian(t, T, p):
    '''
        t: timestep to plot at
        T: number of timesteps
        D: number of spatial dimensions (can be 0)
        Generates a D dimensional moving gaussian function, that moves through the space over T discrete timesteps
        The space it moves through is always a 10x10x..x10 hypercube centered at the origin
        Returns a function f(t, p)
        f(t,p) returns a value at time t < timesteps at position p where p is d-dimensional a row vector
    '''
    # TODO use D
    # atm assuming D = 1
    # first dimension of the distribution is time
    pos = 2*_CUBE_SIZE*(t/T) - _CUBE_SIZE
    # mean
    mu = np.array([pos])
    # stddev
    sigma = np.array([_CUBE_SIZE/4])
    # covariance matrix
    covariance = np.diag(sigma**2)
    # points sampled from the normal distribution

    return 200*multivariate_normal.pdf(p, mean=mu, cov=covariance)

def gen_sensor_data(T, D, N, func): # other params: moving, collaborating, density function...
    '''
        T: number of timesteps
        D: number of spatial dimensions
        sensor_variances: Array of N sensors' variances
        func: function to generate data. f(t, p) that returns true value at time t at position p
        Creates N 'sensors' distributed randomly across a 100x100x..x100 hypercube centred at the origin
        For T discrete timesteps, simulate the capture of data by these sensors using their variances
        Generate a TxN table of each sensor's data at timestep t for all 0<t<T
        Each sensor reading is also associated with a location
    '''
    # TODO currently assuming D = 1
    # we assume variances don't change in time/space
    sensor_variances = uniform(1, 5, N)

    # positions for N sensors across T timesteps (T*N vector)
    # sensors all in the same spot
    #sensor_positions = np.zeros((T,N,1))
    # static but distributed sensors
    poses = _CUBE_SIZE*2*rand(N) - _CUBE_SIZE
    sensor_positions = [[[poses[j]] for j in range(N)] for i in range(T)]

    sensor_data = np.array([[normal(func(i, T, sensor_positions[i][j]), sensor_variances[j]) for j in range(N)] for i in range(T)])
    return sensor_positions, sensor_data

def main():
    #plot_func(10, moving_gaussian)
    sensor_pos, sensor_data = gen_sensor_data(24, 1, 10, moving_gaussian)
    plot_observations(24, moving_gaussian, sensor_pos, sensor_data)

if __name__ == '__main__':
    main()


