"""Functions to analyse stochastic processes"""
import numpy as np


def autocorrelation(realisations: np.ndarray) -> np.ndarray:
    """Calculate the (empirical) autocorrelation function of a process.

     Autocorrelation function is defined as R(s,t) = Mean[X(s)X^*(t)]
    :param realisations: a np.ndarray of shape (#realisations, #time points)
    :return: a (N, N) np.ndarray where the [i,j] entry is the correlation of
    the ith and jth entry in the sample functions.
    """

    N = realisations.shape[1]  # number of time points
    result = np.empty((N, N), dtype=complex)
    for s in range(N):
        for t in range(s + 1):  # it is enough to calculate for t<=s
            rst = np.mean(realisations[:, s] * np.conj(realisations[:, t]))
            result[s, t] = rst
            result[t, s] = np.conj(rst)
    return result
