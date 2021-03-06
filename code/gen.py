
import numpy as np
from scipy.stats import multivariate_normal
from numpy.random import rand, normal, random_integers, uniform

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim


CUBE_SIZE=10


def plot_func(T, func):
    '''
        Assuming 1 dimension, plot func
    '''
    # positions in 2 axis of points to plot on
    # TODO right now this -1 +1 business is a hack to get contour() to plot everything
    x, y = np.mgrid[-1:T+1:1, -CUBE_SIZE:CUBE_SIZE:0.1]

    z = np.array([func(t,T,[i]) for t,i in zip(np.ravel(x), np.ravel(y))])
    # Reshape back to a grid.
    z = z.reshape(x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax.plot_surface(x,y,z)
    #ax.plot_wireframe(x,y,z)
    ax.contour(x,y,z, zdir='x', colors=['orange','red','purple','blue','green'], levels=range(T))

    plt.show()

def _plot_function(ax, T, func, color):
    # positions in 2 axis of points to plot on
    x, y = np.mgrid[-1:T+1:1, -10.0:10.0:0.1]

    z = np.array([func(t,T,[i]) for t,i in zip(np.ravel(x), np.ravel(y))])
    # Reshape back to a grid.
    z = z.reshape(x.shape)

    ax.contour(x,y,z, zdir='x', colors=[color], levels=range(T))

    return ax

def _plot_observations(ax, T, sensor_pos, sensor_data):
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

    ax.scatter(sensor_x, sensor_y, sensor_z, c='b')

    return ax

def plot_observations(T, func, sensor_pos, sensor_data):
    '''
        Assuming 1 dimension, plot func vs sensor data
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax = _plot_function(ax, T, func, 'green')
    _plot_observations(ax, T, sensor_pos, sensor_data)

    plt.show()

def plot_interp_over_time(T, func, sensor_pos, sensor_data, estimates, grid_positions):
    '''
        Assuming 1 dimension, plot func
        Then plot estimates at each grid position on top
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax = _plot_function(ax, T, func, 'green')
    ax = _plot_observations(ax, T, sensor_pos, sensor_data)

    # contours for interpolation estimates instead of points
    '''
    grid_map = {t:{grid_positions[i]:estimates[t][i] for i in range(len(grid_positions))} for t in range(T)}

    # positions in 2 axis of points to plot on
    x = np.array([[t]*len(grid_positions) for t in range(T)])
    y = np.array([[a for a in grid_positions] for i in range(T)])

    z = np.array([grid_map[t][i] for t,i in zip(np.ravel(x), np.ravel(y))])
    # Reshape back to a grid.
    z = z.reshape(x.shape)

    ax.contour(x,y,z, zdir='x', colors=['red'], levels=range(T))
    '''

    # position in time
    est_x = []
    # position in space
    est_y = []
    # estimate
    est_z = []
    # we need T*N data points
    for i in range(T):
        for j in range(len(grid_positions)):
            est_x.append(i)
            est_y.append(grid_positions[j])
            est_z.append(estimates[i][j])

    ax.scatter(est_x, est_y, est_z, c='r')

    plt.show()


def moving_gaussian(t, T, p):
    '''
        t: timestep to plot at
        T: number of timesteps
        D: number of spatial dimensions (can be 0)
        Generates a D dimensional moving gaussian function, that moves through the space over T discrete timesteps
        The space it moves through is always a hypercube centered at the origin
        Returns a function f(t, p)
        f(t,p) returns a value at time t < timesteps at position p where p is d-dimensional a row vector
    '''
    # TODO use D
    # atm assuming D = 1
    # first dimension of the distribution is time
    pos = 2*CUBE_SIZE*(t/T) - CUBE_SIZE
    # mean
    mu = np.array([pos])
    # stddev
    sigma = np.array([CUBE_SIZE/4])
    # covariance matrix
    covariance = np.diag(sigma**2)
    # points sampled from the normal distribution

    return 200*multivariate_normal.pdf(p, mean=mu, cov=covariance)

def gen_sensor_data(T, D, N, func, distr_movement='static', distr_type='even', distr_variance=0.3): # other params: moving, collaborating, density function...
    '''
        T: number of timesteps
        D: number of spatial dimensions
        sensor_variances: Array of N sensors' variances
        func: function to generate data. f(t, p) that returns true value at time t at position p
        static: sensors d
        distr_movement: move each timestep
            - 'static' = no movement
            - 'random' = jump to a random point accordint to distr_type each timestep
            - 'moving' = move a random amount in a random direction each timestep
        distr_type: pattern of sensor distribution
            - 'even' = even intervals across space with some variance
            - 'random' = completely random. if moving,
        distr_variance: variance parameter for even spacing randomness
        Creates N 'sensors' distributed randomly across a hypercube centred at the origin
        For T discrete timesteps, simulate the capture of data by these sensors using their variances
        Generate a TxN table of each sensor's data at timestep t for all 0<t<T
        Each sensor reading is also associated with a location
    '''
    # TODO currently assuming D <= 1
    # we assume variances don't change in time/space
    sensor_variances = uniform(0.5, 5, N)

    # positions for N sensors across T timesteps (T*N vector)
    if D == 0:
        # sensors all in the same spot
        sensor_positions = np.zeros((T,N,1))
    else:
        if distr_movement == 'static':
            if distr_type == 'random':
                poses = uniform(-CUBE_SIZE, CUBE_SIZE, N)
            elif distr_type == 'even':
                poses = np.array([normal(p, distr_variance) for p in np.arange(-CUBE_SIZE, CUBE_SIZE, (CUBE_SIZE*2)/N)])
                print(poses)
            else:
                raise("Not implemented")
            sensor_positions = [[[poses[j]] for j in range(N)] for i in range(T)]
        else:
            raise("Not implemented")

    sensor_data = np.array([[normal(func(i, T, sensor_positions[i][j]), sensor_variances[j]) for j in range(N)] for i in range(T)])
    return sensor_positions, sensor_variances, sensor_data

def main():
    #plot_func(10, moving_gaussian)
    sensor_pos, sensor_vars, sensor_data = gen_sensor_data(24, 1, 10, moving_gaussian)
    plot_observations(24, moving_gaussian, sensor_pos, sensor_data)

if __name__ == '__main__':
    main()


