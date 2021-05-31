import ctypes

import numpy as np
from numba import jit, int64, float32, prange, vectorize
from numba.extending import get_cython_function_address

from statsmodels.api import OLS
from statsmodels.graphics import gofplots
from statsmodels.tools import add_constant
import scipy.stats as scstats

## Descriptive statistics (One-sample)
mean = np.mean

@jit("float32(float32[:])", nopython = True)
def std(a):
    N = len(a)
    correction_factor = float32(np.sqrt(N/(N-1)))
    return np.std(a)*correction_factor

@jit("float32(float32[:])", nopython = True)
def median(a):
    return float32(np.median(a))

@jit("float32(float32[:])", nopython = True)
def skewness(a):
    Da = a - np.mean(a)
    Da2 = Da*Da
    Da3 = Da2*Da
    skew = np.mean(Da3)/np.mean(Da2)**1.5
    return skew    

descr_stat_func = {"mean": mean, "median": median, "stdev": std, "skewness": skewness}


## Two-sample statistics
@vectorize("float32(float32, float32)", nopython = True)
def _A_ij(x1, x2):
    if x1>x2:
        return 1.0
    elif x1 == x2:
        return 0.5
    else:
        return 0.0

@jit("float32(float32[:], float32[:])", nopython = True, parallel = True)
def A_statistic(sample1, sample2):
    N1, N2 = len(sample1), len(sample2)
    A = 0.0
    for i in prange(N1):
        A += np.sum(_A_ij(sample1[i], sample2))
    return A/(N1*N2)

@jit("float32(float32[:], float32[:])", nopython = True, parallel = True)
def KS_statistic(sample1, sample2):
    sample_both = np.concatenate((sample1, sample2))
    cdf1 = np.searchsorted(sample1, sample_both, 'right') / len(sample1)
    cdf2 = np.searchsorted(sample2, sample_both, 'right') / len(sample2)
    ks = np.max(np.abs(cdf1 - cdf2))
    return ks

## Standard normal distribution

_addr_ndtri = get_cython_function_address("scipy.special.cython_special", "ndtri")
_functype_ndtri = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
normal_ppf = _functype_ndtri(_addr_ndtri)

_addr_ndtr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0ndtr")
_functype_ndtr = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
normal_cdf = _functype_ndtr(_addr_ndtr)

## Quantile-Quantile plot

def calculate_QQplot(data1, data2, a = 0):
    def _sample_quantiles(data):
        probplot = gofplots.ProbPlot(np.array(data, dtype = float), a = a)
        return probplot.sample_quantiles
    
    def _match_quantile_probabilities(quantiles1, quantiles2):
        if len(quantiles1) > len(quantiles2):
            quantiles2, quantiles1 = _match_quantile_probabilities(quantiles2, quantiles1)
        else:
            N_obs = len(quantiles1)
            probs = gofplots.plotting_pos(N_obs, a)
            quantiles2 = scstats.mstats.mquantiles(quantiles2, probs)
        
        return quantiles1, quantiles2

    s1, s2 = _sample_quantiles(data1), _sample_quantiles(data2)
    s1, s2 = _match_quantile_probabilities(s1, s2)
    
    linreg_result = OLS(s2, add_constant(s1)).fit()
    s2_fitted = linreg_result.fittedvalues
    r = np.sqrt(linreg_result.rsquared)
    
    return s1, s2, s2_fitted, r