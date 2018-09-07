### Goal
Combine spatial interpolation weighting techniques (e.g. Barnes) with iterative filtering to defeat noisy, broken or collaborating observers.

The claim is that by robustly estimating the variance of the sensors and using these as weights with a spatial interpolation algorithm will improve the accuracy of such an algorithm.

### Notes

It can be viewed similarly to a Machine Learning problem:
- We have a set of data points, each has a feature vector containing:
    - The (continuous) value we want to estimate
    - The (discrete or continuous) location of the value in space
    - The (discrete) time of the measurement
    - The (discrete) identity of the observer who took the measurement
- Given a vector with *just* the time and location, predict the value

Note that we are not given an observer for the value we want to predict; we're trying to predict the 'truest' value by taking a weighted average of the observers.

In other words, we take some sensor readings across space and time, and produce a function that predicts the value at any point in space/time that is bounded by space/time of the data set.

We do this by weighting the observations based on their relative accuracy using iterative filtering.
We should be able to compare our estimate with a maximum likelihood estimate where we know the true variances and values.

1. Use iterative filtering (combined with distance measure etc) to compute weights
2. Apply interpolation/ML algorithm
3. Compare with maximum likelihood version

Also:
- Final function will be inexact; i.e. will not fit observations exactly!
- We may want to know about the certainty of an estimate: e.g. Kriging techniques
- Existing spatial interpolation methods do often account for error/overfitting, but don't seem to account for past observations (in determining accuracy)

### Milestones

1. [x] Create groundwork for estimation, data generation etc.
2. [x] Disregarding the spatial aspect, implement UNSW IF algo and maximum likelihood estimate (MLE)
3. [x] Set up spatial interpolation algorithm and implement Barnes Filtering or another algorithm
4. [x] Apply variance weighting to interpolation algorithm
5. [ ] Apply Maximum Likelihood Estimate to interpolation algorithm and compare with regular version
6. [ ] Apply modified UNSW IF algorithm for estimating variances

(1), (2) and (3) are necessary groundwork.
(4) and (5) if possible, proves the claim that variance weighting may improve spatial interpolation techniques.
(6), if possible, will be a practical method for achieving good variance estimates

### Predictions
- This _should_ work...it's really a matter of weighting the variance prediction algorithm by distance, and applying the variances to the interpolation estimates
- This may only work when there are lots of sensors grouped in the same area

