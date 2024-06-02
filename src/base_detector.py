import numpy as np
import ruptures as rpt
from ruptures.base import BaseCost
from scipy.optimize import curve_fit

__DT = 0.1


def empirical_msd(trajectory, lag, k):
    """Generate empirical MSD for a single lag.

    :param trajectory: numpy.ndarray, list of (x,y) coordinates of a particle
    :param lag: int, time lag
    :param k: int, power of msd
    :return: msd for given lag
    """
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    N = len(x)
    x1 = np.array(x[: (N - lag)])
    x2 = np.array(x[lag:N])
    y1 = np.array(y[: (N - lag)])
    y2 = np.array(y[lag:N])
    c = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) ** k
    r = np.mean(c)
    return r


def generate_empirical_msd(trajectory, mlag, k=2):
    """Generate empirical MSD for a list of lags from 1 to mlag.

    :param trajectory: numpy.ndarray, array of (x,y) coordinates
    :param mlag: int, max time lag for msd
    :param k: int, power of msd
    :return: array of empirical msd
    """
    r = []
    for lag in range(1, mlag + 1):
        r.append(empirical_msd(trajectory, lag, k))
    return np.array(r)


def estimate_with_noise_1(trajectory, mlag, k=2):
    """
    The estimation of diffusion exponent with noise, according to method I from
    Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska,
    "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data"
    Phys. Rev. E 98, 062139 (2018).
    """
    max_number_of_points_in_msd = mlag + 1
    log_msd = np.log(generate_empirical_msd(trajectory, mlag, k))
    log_n = np.array([np.log(i) for i in range(1, max_number_of_points_in_msd)])
    alpha = (
        (max_number_of_points_in_msd + 1) * np.sum(log_n * log_msd)
        - np.sum(log_n * np.sum(log_msd))
    ) / ((max_number_of_points_in_msd + 1) * np.sum(log_n**2) - (np.sum(log_n)) ** 2)
    D = np.exp(log_msd[0]) / 4
    return D, alpha


def estimate_with_noise_3(trajectory, mlag, k=2):
    """
    The estimation of diffusion exponent with noise, according to method III from with n_min fixed from
    Y. Lanoiselée, G. Sikora, A. Grzesiek, D. S. Grebenkov, and A. Wyłomańska,
    "Optimal parameters for anomalous-diffusion-exponent estimation from noisy data"
    Phys. Rev. E 98, 062139 (2018).
    """
    mlag = min(mlag, len(trajectory) - 1)
    empirical_msd = generate_empirical_msd(trajectory, mlag, k)
    n_list = np.array(range(1, mlag + 1))  # POTENCJALNY BLAD w RANGE
    alpha_0 = 1
    D_0 = empirical_msd[0] / 4  # INACZEJ NIŻ W ORYGINALE (BLAD?)
    eps = 0.01

    def msd_fitting(n_list, de, dt, al):
        r = 4 * de * dt**al * (n_list - 1) ** al
        return r

    popt, cov = curve_fit(
        lambda x, D, a: msd_fitting(x, D, __DT, a),
        n_list,
        empirical_msd,
        p0=(D_0, alpha_0),
        bounds=([0, 0], [np.inf, 2]),
        method="dogbox",
        ftol=eps,
    )

    D_est = popt[0]
    alpha_est = popt[1]
    return D_est, alpha_est


def alpha_estim(x):
    _, alpha = estimate_with_noise_3(x, 5)
    return alpha


def D_estim(x):
    D, _ = estimate_with_noise_1(x, 5)
    return D


class MultiEstimCost(BaseCost):
    """Custom cost for exponential signals."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, signal):
        self.signal = signal

    def error(self, start, end):
        half_of_interval = start + int((end - start) / 2)
        first_half = self.signal[start:half_of_interval]
        second_half = self.signal[half_of_interval:end]
        estimates_1 = [
            self.estimators[i](first_half) for i in range(len(self.estimators))
        ]
        estimates_2 = [
            self.estimators[i](second_half) for i in range(len(self.estimators))
        ]
        return sum(
            abs(estimates_1[i] - estimates_2[i]) for i in range(len(estimates_1))
        )


class BaseDetector:

    def __init__(self, window_length=20):
        self.window_length = window_length
        self.algo = rpt.Window(
            width=window_length,
            custom_cost=MultiEstimCost([D_estim, alpha_estim]),
            jump=1,
        )
        self.penalty = 2 * np.log(window_length) * 0.12**2

    def predict(self, trajectory):
        self.algo.fit(trajectory)
        change_points = self.algo.predict(epsilon=self.penalty)
        return change_points
