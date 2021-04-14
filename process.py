"""Implementations of statistical processes"""
from typing import Tuple, Union
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


def ornstein_uhlenbeck(T: float, dt: float,
                       x_0: Union[float, np.ndarray] = None, gamma: float = 1.0,
                       sigma: float = 0.3, size: int = 1, is_complex = False) -> Tuple[
                       np.ndarray, np.ndarray]:
    """Sample the real Ornstein-Uhlenbeck process.

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
            x = realpart + imagpart*1j
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
