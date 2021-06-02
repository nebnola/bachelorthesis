import numpy as np
import matplotlib.pyplot as plt
import fit


def viz_fit(fit_result: fit.FitResult, mode='compar', T=None, **kwargs):
    """Plot the result of a fit

    :param fit_result: a fit.FitResult instance
    :param mode: One of 'compar' (default) or 'diff'. If set to 'compar' shows the target function and the fitted
    :param T: the maximum time up to which plot is drawn, optional
    function side by side. If 'diff' plots the absulute difference
    """
    if T is None:
        T = max(fit_result.t)
    t = np.linspace(0, T, 500)
    vals = fit_result.fit_values(t)
    target_vals = fit_result.target_values(t)
    if mode == 'compar':
        fig = plt.figure()
        re, im = fig.subplots(1, 2)
        re.set_title('real part')
        im.set_title('imaginary part')
        re.plot(t, vals.real, label='target')
        im.plot(t, vals.imag, label='target')
        re.plot(t, target_vals.real, label='fit')
        im.plot(t, target_vals.imag, label='fit')
        plt.legend()
        # return fig
    elif mode == 'diff':
        diff = np.abs(vals - target_vals)
        plt.plot(t, diff, **kwargs)
        plt.legend()
