import numpy as np
from gen import gen_sensor_data, moving_gaussian, plot_observations

def predict_unsw_v1(sensor_data):
    return predict_unsw(sensor_data, 1)

def predict_unsw_v2(sensor_data):
    return predict_unsw(sensor_data, 2)

def predict_unsw(sensor_data, v):
    '''
        Predict true values using of UNSW's iterative filtering algorithm
        version 1 uses the k'th sensor's estimate of the value when computing likelihoods
        version 2 uses the estimate, but still using the kth's estimated variance
    '''
    T = len(sensor_data)
    N = len(sensor_data[0])

    # use longdouble for everything
    sensor_data = np.longdouble(sensor_data)

    # initial estimate for each timestep is just the mean of the data
    estimate = np.longdouble(
            np.array([sum(timestep)/N for timestep in sensor_data])
            )
    # we estimate all sensors to have the same variance initially
    # we just average all estimated variances based on the initial estimate
    initial_var = np.sum([np.sum([(sensor_data[t][j] - estimate[t])**2 for t in range(T)])/(T-1) for j in range(N)])/N
    variances = np.longdouble(np.array([initial_var]*N))

    eps = 10e-8
    diff = 1
    it = 0
    while diff > eps:
        # update estimate
        old_estimate = estimate

        # likelihood of each sensor being correct, by taking normalised likelihoods across all k sensors
        likelihoods = np.array([
            np.power(np.product([
                    np.exp(
                        ((-1/(T*2*variances[k]))*np.sum([(sensor_data[t][j] - (sensor_data[t][k] if v==1 else estimate[t]))**2 for t in range(T)]))
                        )/np.sqrt(2*np.pi*variances[k])
                for k in range(N)]), 1/N)
            for j in range(N)])
        #print("likelihoods:",likelihoods)

        # next estimate is just the MLE using likelihoods instead of 1/var
        sum_of_likelihoods = np.sum(likelihoods)
        estimate = np.array([sum([((likelihoods[j])/sum_of_likelihoods)*sensor_data[i][j] for j in range(N)])for i in range(T)])

        # compute new variances using new estimate of true value
        variances = np.array([np.sum([(sensor_data[t][j] - estimate[t])**2 for t in range(T)])/(T-1) for j in range(N)])
        #print("variances:",variances)

        # calculate rms of old and new estimates to see if they're converging
        diff = np.sqrt(sum([(estimate[i] - old_estimate[i])**2 for i in range(T)]))
        #print("diff:",diff)

        it += 1

    #print("converged after",it,"iterations")

    # final estimate is just MLE using the predicted variances
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
    N = 10  # no. sensors
    # we use D = 0; All sensors will be in same location
    sensor_pos, sensor_vars, sensor_data = gen_sensor_data(T, 0, N, moving_gaussian)
    #plot_observations(T, moving_gaussian, sensor_pos, sensor_data)

    # first, MLE for optimal
    mle_estimate = predict_MLE(sensor_data, sensor_vars)
    # moving_gaussian is the true value, so we take rms with that
    rms_error = np.sqrt(sum([(mle_estimate[i] - moving_gaussian(i, T, [0]))**2 for i in range(T)]))
    print("mle error:",rms_error)

    # now, UNSW iterative filtering
    unsw_v1_estimate = predict_unsw_v1(sensor_data)
    rms_error = np.sqrt(sum([(unsw_v1_estimate[i] - moving_gaussian(i, T, [0]))**2 for i in range(T)]))
    print ("unsw algorithm v1 error:",rms_error)

    unsw_v2_estimate = predict_unsw_v2(sensor_data)
    rms_error = np.sqrt(sum([(unsw_v2_estimate[i] - moving_gaussian(i, T, [0]))**2 for i in range(T)]))
    print ("unsw algorithm v2 error:",rms_error)

if __name__=='__main__':
    main()

