import numpy as np
from gen import gen_sensor_data, moving_gaussian

def predict_unsw(sensor_data):
    T = len(sensor_data)
    N = len(sensor_data[0])

    # initial estimate for each timestep is just the mean
    means = np.array([sum(timestep)/N for timestep in sensor_data])
    # initial estimate of each sensor's variance is based on those means
    variances = np.array([sum([(sensor_data[i][j] - means[i])**2 for i in range(T)])/(T-1) for j in
        range(N)])
    print(means)
    print(variances)


    # TODO rest of algorithm

    #final_est
    #return final_est

def predict_MLE(sensor_data, sensor_vars):
    T = len(sensor_data)
    N = len(sensor_data[0])

    sum_of_vars = sum([1/var for var in sensor_vars])
    mle_estimate = np.array([sum([((1/sensor_vars[j])/sum_of_vars)*sensor_data[i][j] for j in range(N)])for i in range(T)])
    return mle_estimate


def main():
    T = 24  # no. measurements per sensor
    N = 20  # no. sensors
    # we use D = 0; All sensors will be in same location
    sensor_pos, sensor_vars, sensor_data = gen_sensor_data(T, 0, N, moving_gaussian)

    # first, MLE for optimal
    mle_estimate = predict_MLE(sensor_data, sensor_vars)
    # moving_gaussian is the true value, so we take rms with that
    mle_rms_error = np.sqrt(sum([(mle_estimate[i] - moving_gaussian(i, T, [0]))**2 for i in range(T)]))
    print("mle error:",mle_rms_error)

    # now, UNSW iterative filtering TODO
    predict_unsw(sensor_data)
    print ("true variances:",sensor_vars)

if __name__=='__main__':
    main()

