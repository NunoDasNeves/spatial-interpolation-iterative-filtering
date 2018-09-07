import numpy as np
from gen import gen_sensor_data, moving_gaussian, plot_interp_over_time, CUBE_SIZE


def barnes_interp(sensor_readings, sensor_pos, R=1.0, D=1, d_x=0.3, gamma=0.2):
    '''
        sensor_readings: readings from N sensors
        sensor_pos: vector positions of N sensors
        R: max distance sensors to consider
        D: number of spatial dimensions
        d_x: spacing
        gamma: smoothing parameter
        Predict true value at all points within hypercube using Barnes interpolation
    '''
    # TODO everything assumes D = 1 right now!

    N = len(sensor_readings)

    # area being predicted over
    area = CUBE_SIZE**D
    # characteristic data spacing
    d_n = (area**(1/2))*((1+N**(1/2))/(N-1))
    # falloff parameter
    K = 5.052*(2*d_n/np.pi)**2

    # use longdouble for everything
    sensor_readings = np.longdouble(sensor_readings)
    sensor_pos = np.longdouble(np.array(sensor_pos).reshape(N))

    # get uniformly spaced grid positions we're going to interpolate over
    # TODO currently assumes D = 1
    grid_positions = np.array([p for p in np.arange(-CUBE_SIZE, CUBE_SIZE, d_x)])

    grid_estimates = []         # store the estimates for each point
    closest_sensor_lists = []   # store a list of sensor readings & distances for each point's closest sensors

    # do initial estimate for all points p
    for p in grid_positions:
        # get distance of all sensors to p
        sensor_dist = list(map(lambda t:np.abs(t[0]-t[1]), zip([p]*N, sensor_pos)))
        # find all sensors within R distance of p to produce [(val, dist), (val, dist) ...]
        # zip and unzip back into two lists
        # TODO this only works if there are sensors within R of p!
        #sensor_readings, sensor_dist = zip(*[(val, d) for val,d in zip(sensor_readings, sensor_dist) if d < R])

        weights = np.array([np.exp(-d**2/K) for d in sensor_dist])
        weights_sum = np.sum(weights)
        grid_estimates.append(np.sum([w*m for w,m in zip(weights, sensor_readings)])/weights_sum)
        closest_sensor_lists.append(list(zip(sensor_readings, sensor_dist, sensor_pos)))

    # map of positions to estimates for convenience
    grid_map = {pos:val for pos,val in zip(grid_positions, grid_estimates)}

    # we use linear interpolation to compute estimates for the points at the sensor positions
    # we have to do something like this to account for the fact that the sensor positions are not on our grid
    # TODO this could be faster, or we could possibly do it as part of the previous step
    for i in range(N):
        # find the two closest grid positions (i.e. one before and after) with a linear scan
        lower_pos = grid_positions[0]
        upper_pos = grid_positions[-1]
        for p in grid_positions:
            if p > lower_pos and p < sensor_pos[i]:
                lower_pos = p
            if p < upper_pos and p > sensor_pos[i]:
                upper_pos = p
        # ratio is a value scaled between 0 and 1 denoting how far between
        # the lower and upper grid lines the sensor is
        ratio = (sensor_pos[i] - lower_pos)/d_x
        # linear interpolation, stored in grid_map for easy access later
        grid_map[sensor_pos[i]] = ((1-ratio)*grid_map[lower_pos] + ratio*grid_map[upper_pos])

    # iterate once for final estimate across all grid points
    final_estimates = []

    for k in range(len(grid_positions)):
        # note these weights have a gamma
        weights = np.array([np.exp(-(d)**2/(gamma*K)) for _,d,p in closest_sensor_lists[k]])
        # for this grid position, sum up weights to use in denominator
        weights_sum = np.sum(weights)
        # add to the estimate a weighted difference between estimates and actual sensor values
        final_estimates.append(grid_estimates[k] + np.sum([(val - grid_map[p])*w for w,val,_,p in
            zip(weights, *zip(*closest_sensor_lists[k]))])/weights_sum)

    return final_estimates


def main():
    T = 24  # no. measurements per sensor
    N = 20  # no. sensors
    # we use D = 1; Sensors will be spread out across 1D space
    sensor_pos, sensor_vars, sensor_data = gen_sensor_data(T, 1, N, moving_gaussian)

    d_x = 0.3
    grid_positions = np.array([p for p in np.arange(-CUBE_SIZE, CUBE_SIZE, d_x)])
    estimates = []
    for t in range(T):
        est = barnes_interp(sensor_data[t], sensor_pos[t], d_x=d_x)
        # TODO precision, recall etc
        #rms_error = np.sqrt(sum([(est[i] - moving_gaussian(t, T, (grid_positions[i],)))**2 for i in range(len(grid_positions))]))
        #print("barnes rms error at time {}: {}".format(t,rms_error))
        estimates.append(est)

    plot_interp_over_time(T, moving_gaussian, sensor_pos, sensor_data, estimates, grid_positions)


if __name__=='__main__':
    main()

