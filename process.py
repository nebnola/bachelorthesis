"""Implementations of statistical processes"""
from typing import Tuple, Union
from numbers import Complex, Real
import numpy as np


def wiener_increment(dt: float, size: int = 1, is_complex: bool = False) -> np.ndarray:
    """Return increments dW of a standard Wiener process.

    It's just a normal distribution with standard deviation sqrt(dt). If size
    is given, return the corresponding number of independent increments
    (useful for simulating many processes at once). If complex is True, return
    the increment of the complex Wiener process, defined as two independent
    Wiener process in the real and imaginary part.
    :param dt: time increment dt
    :param size: number of desired increments
    :param is_complex: specify if increment should be of the complex or real
    Wiener
    process
    :return: corresponding increment(s) of a Wiener process sample function
    """
    if not is_complex:
        return np.random.normal(0, np.sqrt(dt), size)
    else:
        return np.random.normal(0, np.sqrt(dt), size) + np.random.normal(0, np.sqrt(dt), size) * 1j


def wiener(T: float, dt: float, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the standard Wiener process.

    :param T: Duration of the sample function(s)
    :param dt: Desired step size. Is not accurate but the real step size
    is such that it fits evenly into T.
    :param size: number of sample functions to be generated
    :return: Tuple (t, sample) where t is a np.ndarray of time points and
    sample are the sample functions. sample is of shape (size, int(T/dt)+1),
    so axis 0 corresponds to the different samples and axis 1 to time.
    """
    N = int(T / dt) + 1  # number of time steps
    real_dt = T / (N - 1)
    t = np.linspace(0, T, N)
    samples = np.empty((size, N))  # results will go in here
    x = np.full(size, 0)  # start values
    for i in range(N):
        samples[:, i] = x
        x = x + wiener_increment(real_dt, size)

    return t, samples


def ornstein_uhlenbeck_naive(T: float, dt: float, x_0: Union[float, np.ndarray] = None,
                             gamma: Union[float, complex] = 1.0,
                             sigma: float = 0.3, size: int = 1, is_complex=False) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the Ornstein-Uhlenbeck process. Naive implementation, not exact but works for very small dt.

    It follows the SDE dX(t) = -gamma*X(t)*dt + sigma*dW(t)
    :param T: Duration of the sample function(s)
    :param dt: Desired step size. Is not accurate but the real step size
    is such that it fits evenly into T.
    :param x_0: The starting point X(0) of the process. Can be a number or a
     np.ndarray (in which case it must be of length size)
     By default, sampled independently according to the stationary distribution.
    :param gamma: "stiffness" of the process
    :param sigma: "randomness" of the process
    :param size: number of sample functions to be generated
    :param is_complex: specify if complex Ornstein-Uhlenbeck process is meant.
    In that case, gamma can be a complex number.
    :return: Tuple (t, sample) where t is a np.ndarray of time points and
    sample are the sample functions. sample is of shape (size, int(T/dt)+1),
    so axis 0 corresponds to the different samples and axis 1 to time.
    """
    N = int(T / dt) + 1  # number of time steps
    real_dt = T / (N - 1)
    t = np.linspace(0, T, N)  # time points
    if not is_complex:
        samples = np.empty((size, N))  # result will go here
    else:
        samples = np.empty((size, N), dtype=complex)

    # initialise process
    if x_0 is None:  # standard behaviour is to sample from stationary
        # distribution
        if not is_complex:
            x = np.random.normal(0, sigma / (2 * gamma) ** 0.5, size)
        else:
            realpart = np.random.normal(0, sigma / (2 * gamma.real) ** 0.5, size)
            imagpart = np.random.normal(0, sigma / (2 * gamma.real) ** 0.5, size)
            x = realpart + imagpart * 1j
    else:
        x_0 = np.asarray(x_0)
        if x_0.shape == ():  # if x_0 is simply a number
            x_0 = np.full(size, x_0)
        elif x_0.shape != (size,):
            raise ValueError("Shape of x_0 incompatible with selected size")
        x = x_0

    for i in range(N):
        samples[:, i] = x  # write into result array
        w = sigma * wiener_increment(real_dt, size, is_complex)
        x = x - gamma * x * real_dt + w
    return t, samples


def ornstein_uhlenbeck(T: Real, dt: Real, theta: Complex, sigma: Real, size: int = 1,
                       x_0: Union[Complex, np.ndarray] = None,
                       is_complex: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the real or complex Ornstein-Uhlenbeck process

    It follows the SDE dX(t) = -theta*X(t)*dt + sigma*dW(t)
    :param T: Duration of the sample function(s)
    :param dt: Desired step size. Is not accurate but the real step size is such that it fits evenly into T.
    :param theta: "stiffness" of the process
    :param sigma: "noisiness" of the process
    :param size: number of independent sample functions to be generated
    :param x_0: The starting point X(0) of the process. Can be a number or a np.ndarray (in which case it must be of
    length size) By default, sampled independently according to the stationary distribution.
    :param is_complex: specify if Ornstein-Uhlenbeck process should be complex. In that case, theta can be a complex
    number.
    :return: Tuple (t, sample) where t is a np.ndarray of time points and sample are the sample functions. sample is
    of shape (size, int(T/dt)+1), so axis 0 corresponds to the different samples and axis 1 to time.
    """

    N = int(T / dt) + 1  # number of time steps
    real_dt = T / (N - 1)
    t = np.linspace(0, T, N)  # time points
    if not is_complex:
        samples = np.empty((size, N))  # result will go here
    else:
        samples = np.empty((size, N), dtype=complex)

    if x_0 is None:  # standard behaviour is to sample from stationary distribution
        if not is_complex:
            x = np.random.normal(0, sigma / (2 * theta) ** 0.5, size)
        else:
            realpart = np.random.normal(0, sigma / (2 * theta.real) ** 0.5, size)
            imagpart = np.random.normal(0, sigma / (2 * theta.real) ** 0.5, size)
            x = realpart + imagpart * 1j
    else:
        x_0 = np.asarray(x_0)
        if x_0.shape == ():  # if x_0 is simply a number
            x_0 = np.full(size, x_0)
        elif x_0.shape != (size,):
            raise ValueError("Shape of x_0 incompatible with selected size")
        x = x_0

    # let's only compute this once:
    factor = sigma * np.sqrt((1 - np.exp(-2 * theta.real * real_dt)) / (2 * theta.real))

    for i in range(N):
        samples[:, i] = x  # write into result array
        if is_complex:
            noise = factor * (np.random.normal(0, 1, size) + 1j * np.random.normal(0, 1, size))
        else:
            noise = factor * np.random.normal(0, 1, size)
        x = np.exp(-theta * real_dt) * x + noise

    return t, samples


def multivariate_ou_naive(T: Real, dt: Real, dim: int, lambd: np.ndarray, B: np.ndarray, size: int = 1) -> Tuple[
                          np.ndarray, np.ndarray]:
    """
    Multivariate complex Ornstein-Uhlenbeck process with diagonal matrix A, naive implementation

    The multivariate Ornstein-Uhlenbeck process follows the stochastic differential vector equation
    dX(t) = -A*X(t)dt + B*dW(t),
    where W(t) is the complex dim-dimensional Wiener process, and A and B are (dim x dim) matrices. The entries of W(
    t) are realised with independent real and imaginary parts scaled by 1/sqrt(2), so that its
    autocorrelation function is Mean(W^*(s) W(t)) = min(s,t)
    :param T: Duration of the sample function(s)
    :param dt: Desired step size. Is not accurate but the real step size is such that it fits evenly into T.
    :param dim: Dimensionality of the process
    :param lambd: array of length dim such that A = diag(lambd[0], ..., lambd[dim-1]). Entries should only have positive
    real parts, although this is not checked
    :param B: Matrix B of the differential equation
    :param size: Number of independent samples to be drawn.
    :return: Tuple (t, samples)
    t is an array of time points.
    samples is the array of samples of shape (size, int(T/dt) + 1, dim)
    Axis 0 represents the different samples, axis 1 represents the different time points and axis 2 corresponds to the
    entries of the vector
    """
    # matrix A
    A = np.diag(lambd)

    N = int(T / dt) + 1  # number of time steps
    real_dt = T / (N - 1)
    t = np.linspace(0, T, N)  # time points

    # results array: 0th axis: different samples, 1st axis: time, 2nd axis: dim-dimensional vector
    samples = np.empty((size, N, dim), dtype=complex)

    # covariance matrix of stationary distribution:
    cov = B @ B.transpose().conj() / (np.tile(lambd, (dim, 1)).transpose() + np.tile(lambd.conj(), (dim, 1)))
    # if Z=X+iY is stationary dist. then these are the covariances of X and Y:
    xx = yy = cov.real / 2
    xy = - cov.imag / 2
    yx = cov.imag / 2
    realcov = np.block([[xx, xy], [yx, yy]])

    # sample initial value:
    realimag = np.random.multivariate_normal(np.zeros(2 * dim), realcov, size)
    x = realimag[:, :dim] + 1j * realimag[:, dim:]
    x = np.expand_dims(x, axis=2)  # necessary so that simultaneous matrix multiplication of vectors works.
    for i in range(N):
        samples[:, i, :] = np.squeeze(x)
        noise = np.sqrt(real_dt / 2) * (np.random.normal(0, 1, (size, dim)) + 1j * np.random.normal(0, 1, (size, dim)))
        x = x - A @ x * real_dt + B @ np.expand_dims(noise, axis=2)

    return t, samples

