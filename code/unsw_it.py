import numpy as np
from gen import gen_sensor_data, moving_gaussian

def predict_unsw(sensor_data):
    T = len(sensor_data)
    N = len(sensor_data[0])

    # initial estimate for each timestep is just the mean of the data
    old_estimate = np.array([sum(timestep)/N for timestep in sensor_data])
    # initial estimate of each sensor's variance is based on those means
    variances = np.array([np.sum([(sensor_data[i][j] - old_estimate[i])**2 for i in range(T)])/(T-1) for j in range(N)])
    print("initial estimated variances:",variances)
    # TODO fix

    eps = 0.001
    diff = 1
    it = 0
    while diff > eps:
        it += 1
        # likelihood of each sensor being correct
        likelihoods = np.array([
            np.product([
                np.power((2*np.pi*variances[k]),T/2)*np.exp(
                    -np.sum([(sensor_data[i][j] - sensor_data[i][k])**2 for i in range(T)])/(2*variances[k]))
                for k in range(N)])
            for j in range(N)])

        # next estimate is just the MLE using likelihoods instead of 1/var
        sum_of_likelihoods = np.sum(likelihoods)
        new_estimate = np.array([sum([((likelihoods[j])/sum_of_likelihoods)*sensor_data[i][j] for j in range(N)])for i in range(T)])

        # compute new variances using new estimate of true value
        variances = np.array([np.sum([(sensor_data[i][j] - new_estimate[i])**2 for i in range(T)])/(T-1) for j in range(N)])

        # calculate rms of old and new estimates to see if they're converging
        diff = np.sqrt(sum([(old_estimate[i] - new_estimate[i])**2 for i in range(T)]))
        print("diff:",diff)

        # update estimate
        old_estimate = new_estimate

    print("converged after",it,"iterations")

    print("estimated variances:",variances)

    # final estimate is just MLE with predicted variances
    sum_of_vars = sum([1/var for var in variances])
    final_est = np.array([sum([((1/variances[j])/sum_of_vars)*sensor_data[i][j] for j in range(N)])for i in range(T)])

    return final_est

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
    unsw_estimate = predict_unsw(sensor_data)
    print ("true variances:",sensor_vars)
    unsw_rms_error = np.sqrt(sum([(unsw_estimate[i] - moving_gaussian(i, T, [0]))**2 for i in range(T)]))
    print ("unsw algorithm error:",unsw_rms_error)

if __name__=='__main__':
    main()

