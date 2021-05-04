"""Fit generalised Ornstein-Uhlenbeck processes so that their autocorrelation function approaches a given desired
function. The generalised process of dimension N is given by a N-dimensional vector λ and an NxN-matrix B.
Since only the product adjugate(B)*B matters, in general it can be assumed that B is an upper triagonal matrix.
In order to perform the fit, we need a real parameter vector, so we take the real and imaginary parts of all the entries
of λ and B and put them into a vector, resulting in a 2(N + N*(N+1)/2)-dimensional problem. We use
matrices_from_params() to convert from this parameter vector into λ and B"""
from typing import Callable
import numpy as np
from scipy import optimize
import process


def example_autocorrelation(t):
    """Some autocorrelation function we want to achieve"""
    s = 0.5
    return 1 / ((1 + t * 1j) ** (s + 1))


def matrices_from_params(param, dim):
    """Calculate the vector λ (shape (dim,)) and the matrix B (shape (dim,dim)) for a given parameter vector

    The parameter vector has length 2*(dim + dim*(dim+1)/2), where the first dim entries are the real part of the vector
    λ, the next dim*(dim+1)/2 entries are the real parts of the elements of the upper triangular matrix B and the
    imaginary parts follow in the same manner"""
    length = 2 * (dim + dim * (dim + 1) / 2)
    if len(param) != length:
        raise ValueError("Parameter vector needs to be of size 2*(dim + dim*(dim+1)/2),"
                         "with dim entries for λ and dim*(dim+1)/2 entries for B")
    real = param[:int(length / 2)]
    imag = param[int(length // 2):]
    lambdas = real[:dim] + 1j * imag[:dim]
    Bentries = real[dim:] + 1j * imag[dim:]
    B = np.zeros((dim, dim), dtype=complex)
    B[np.triu_indices(dim)] = Bentries
    return np.asarray(lambdas), B


def residuals_ou(param, target_fun, t, dim):
    """Calculate deviation between an autocorrelation function given by target_fun and that of a generalised O-U process

    The O-U process is characterised by the parameter vector param, which encodes λ and B in the way described by
    matrices_from_params().
    :param param: parameter vector describing the generalised O-U process. Has to be of length 2*(dim+dim*(dim+1)/2)
    :param target_fun: Function for the autocorrelation function that should be approximated. Has to accept and return
    np.arrays of time points
    :param t: Array of time points at which both autocorrelation functions are evaluated
    :param dim: Dimensionality of the generalised O-U process
    :return: Difference of the given autocorrelation function and that of the O-U-process, evaluated at points t."""
    lambdas, b = matrices_from_params(param, dim)
    return np.abs(target_fun(t) - process.general_ou_autoc(lambdas, b, t))


def fit_general_ou(target_fun: Callable, t: np.ndarray, dim: int):
    """Fit the generalized Ornstein-Uhlenbeck process to the desired function example_autocorrelation.

    Mostly a convenience function
    :param target_fun: function that the autocorrelation should be fit to. Has to accept and return np.arrays
    :param t: time points at which the functions are evaluated
    :param dim: Number of dimensions that the generalized O-U process should have
    :return: The optimize.OptimizeResult object that results from the fit"""
    x0 = np.ones(int(2 * (dim + dim * (dim + 1) / 2)))  # simply set all parameters to 1
    o = optimize.least_squares(residuals_ou, x0, kwargs=dict(target_fun=target_fun,
                                                             t=t,
                                                             dim=dim))
    return o
