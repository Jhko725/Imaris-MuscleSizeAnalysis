
import numpy as np
from numba import jit, float32, prange, int64

from statistics import normal_cdf, normal_ppf

from numpy.random import rand

## API level functions 
def bootstrap_confidence_interval(sample, stat_func, B = 10000, confidence_level = 0.95, CI_algorithm = "basic", random_seed = 10):
    sample_stat = stat_func(sample)
    bootstrap_func = bootstrap_factory(stat_func)
    bootstrap_dist = bootstrap_func(sample, B, random_seed)

    if CI_algorithm == "basic":
        CI = _calculate_CI_basic(sample_stat, bootstrap_dist, confidence_level)

    elif CI_algorithm == "BCa":
        jackknife_func = jackknife_factory(stat_func)
        jackknife_dist = jackknife_func(sample)
        CI = _calculate_CI_BCa(sample_stat, bootstrap_dist, jackknife_dist, confidence_level)
    
    return sample_stat, CI

def bootstrap_confidence_interval_2samp(sample1, sample2, stat_func_2samp, B = 10000, confidence_level = 0.95, CI_algorithm = "basic", random_seed = 10):
    sample_stat = stat_func_2samp(sample1, sample2)
    bootstrap_func_2samp = bootstrap_factory_2samp(stat_func_2samp)
    bootstrap_dist = bootstrap_func_2samp(sample1, sample2, B, random_seed)

    if CI_algorithm == "basic":
        CI = _calculate_CI_basic(sample_stat, bootstrap_dist, confidence_level)

    elif CI_algorithm == "BCa":
        jackknife_func_2samp_1, jackknife_func_2samp_2 = jackknife_factory_2samp(stat_func_2samp)
        jackknife_dist1 = jackknife_func_2samp_1(sample1, sample2)
        jackknife_dist2 = jackknife_func_2samp_2(sample1, sample2)
        CI = _calculate_CI_BCa_2samp(sample_stat, bootstrap_dist, jackknife_dist1, jackknife_dist2, confidence_level)
    
    return sample_stat, bootstrap_dist, CI

## Lower level functions

def bootstrap_factory(stat_func):
    # stat_func must be a one-sample statistical function with signature float32[:] -> float32
    # it must also be compatible with numpy & numba
    @jit("float32[:](float32[:], int64, int64)", nopython = True, parallel = True)
    def bootstrap_func(sample, B, random_seed):
        np.random.seed(random_seed)
        bootstrap_dist = []
        for i in prange(B):
            resample = np.random.choice(sample, size = sample.shape)
            bootstrap_dist.append(stat_func(resample))
        return np.array(bootstrap_dist)

    return bootstrap_func

def jackknife_factory(stat_func):
    # same requirements for the stat_func as in "bootstrap_factory" apply
    @jit("float32[:](float32[:])", nopython = True, parallel = True)
    def jackknife_func(sample):
        jackknife_dist = []
        for i in prange(len(sample)):
            jackknife_sample = np.concatenate((sample[:i], sample[i+1:]))
            jackknife_dist.append(stat_func(jackknife_sample))
        return np.array(jackknife_dist)

    return jackknife_func


def bootstrap_factory_2samp(stat_func_2samp):
    # stat_func must be a one-sample statistical function with signature float32[:] -> float32
    # it must also be compatible with numpy & numba
    @jit("float32[:](float32[:], float32[:], int64, int64)", nopython = True, parallel = True)
    def bootstrap_func_2samp(sample1, sample2, B, random_seed):
        np.random.seed(random_seed)
        bootstrap_dist = []
        for i in prange(B):
            resample1 = np.random.choice(sample1, size = sample1.shape)
            resample2 = np.random.choice(sample2, size = sample2.shape)
            bootstrap_dist.append(stat_func_2samp(resample1, resample2))
        return np.array(bootstrap_dist)

    return bootstrap_func_2samp


def jackknife_factory_2samp(stat_func_2samp):
    @jit("float32[:](float32[:], float32[:])", nopython = True, parallel = True)
    def jackknife_func_2samp_1(sample1, sample2):
        # jackknifes along the first sample
        jackknife_dist = []
        for i in prange(len(sample1)):
            jackknife_sample1 = np.concatenate((sample1[:i], sample1[i+1:]))
            jackknife_dist.append(stat_func_2samp(jackknife_sample1, sample2))
        return np.array(jackknife_dist)

    @jit("float32[:](float32[:], float32[:])", nopython = True, parallel = True)
    def jackknife_func_2samp_2(sample1, sample2):
        # jackknifes along the first sample
        jackknife_dist = []
        for i in prange(len(sample1)):
            jackknife_sample2 = np.concatenate((sample2[:i], sample2[i+1:]))
            jackknife_dist.append(stat_func_2samp(sample1, jackknife_sample2))
        return np.array(jackknife_dist)

    return jackknife_func_2samp_1, jackknife_func_2samp_2


@jit("UniTuple(float32, 2)(float32, float32[:], float32)", nopython = True)
def _calculate_CI_basic(sample_stat, bootstrap_dist, confidence_level):
    half_alpha = 0.5*(1-confidence_level)
    CI_differences = np.quantile(bootstrap_dist - sample_stat, (half_alpha, 1-half_alpha))
    CI = (sample_stat - CI_differences[1], sample_stat - CI_differences[0])
    return CI


@jit("UniTuple(float32, 2)(float32[:])", nopython = True)
def _scaled_moments(a):
    n = len(a)
    Da = a - np.mean(a)
    Da2 = Da*Da
    m2 = np.mean(Da2)
    Da3 = Da2*Da
    m3 = np.mean(Da3)
    return m2/n**2, m3/n**3


@jit("float32(float32, float32[:])", nopython = True)
def _BCa_bias(sample_stat, bootstrap_dist):
    bias_score = np.sum(bootstrap_dist < sample_stat)/len(bootstrap_dist)
    return normal_ppf(bias_score)


@jit("float32(float32[:])", nopython = True)
def _BCa_acceleration(jackknife_dist):
    # Note that the ordering of the theta_\mean and \theta_i are reversed
    # between the original definition and the "skewness" function, 
    # which leads to a discrepancy by a factor of -1; thus the - sign
    l_sq, l_cb = _scaled_moments(jackknife_dist)
    return -(1/6)*l_cb/l_sq**1.5


@jit("float32(float32[:], float32[:])", nopython = True)
def _BCa_acceleration_2samp(jackknife_dist1, jackknife_dist2):
    l1_sq, l1_cb = _scaled_moments(jackknife_dist1)
    l2_sq, l2_cb = _scaled_moments(jackknife_dist2)
    return -(1/6)*(l1_cb + l2_cb)/(l1_sq + l2_sq)**1.5


@jit("UniTuple(float32, 2)(float32, float32[:], float32[:], float32)", nopython = True)
def _calculate_CI_BCa(sample_stat, bootstrap_dist, jackknife_dist, confidence_level):
    half_alpha = 0.5*(1-confidence_level)

    z0 = _BCa_bias(sample_stat, bootstrap_dist)
    zL = z0 + normal_ppf(half_alpha)
    zU = z0 + normal_ppf(1-half_alpha)

    a = _BCa_acceleration(jackknife_dist)
    
    alpha1 = normal_cdf( z0 + zL/(1-a*zL) )
    if a*zU<1:
        alpha2 = normal_cdf( z0 + zU/(1-a*zU) )
    else:
        alpha2 = 1.0

    CI = np.quantile(bootstrap_dist, (alpha1, alpha2))
    return (CI[0], CI[1])


@jit("UniTuple(float32, 2)(float32, float32[:], float32[:], float32[:], float32)", nopython = True)
def _calculate_CI_BCa_2samp(sample_stat, bootstrap_dist, jackknife_dist1, jackknife_dist2, confidence_level):
    half_alpha = 0.5*(1-confidence_level)

    z0 = _BCa_bias(sample_stat, bootstrap_dist)
    zL = z0 + normal_ppf(half_alpha)
    zU = z0 + normal_ppf(1-half_alpha)

    a = _BCa_acceleration_2samp(jackknife_dist1, jackknife_dist2)
    
    alpha1 = normal_cdf( z0 + zL/(1-a*zL) )
    if a*zU<1:
        alpha2 = normal_cdf( z0 + zU/(1-a*zU) )
    else:
        alpha2 = 1.0

    CI = np.quantile(bootstrap_dist, (alpha1, alpha2))
    return (CI[0], CI[1])
