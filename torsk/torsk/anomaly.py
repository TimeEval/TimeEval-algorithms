import numpy as np
from scipy import special


def cumulative_distribution(x):
    return special.erf(x / 2. ** .5)


def qfunction(x):
    return 1 - cumulative_distribution(x)


def sliding_score(error, small_window, large_window):
    scores = np.empty(error.shape)
    lw_mu = np.zeros_like(scores)
    lw_std = np.zeros_like(scores)
    sw_mu = np.zeros_like(scores)

    lw_mu[0] = error[0]
    lw_std[0] = error[:2].std(axis=0)
    sw_mu[0] = error[0]

    for i in range(1, error.shape[0]):
        lw_start = max(0, i - large_window + 1)
        sw_end = min(i + small_window, error.shape[0])

        lw_err = error[lw_start:i]
        sw_err = error[i:sw_end]

        lw_mu[i] = lw_err.mean(axis=0)
        lw_std[i] = lw_err.std(axis=0)
        sw_mu[i] = sw_err.mean(axis=0)

        x = np.maximum(0, sw_mu[i] - lw_mu[i]) / (lw_std[i] + (np.abs(lw_std[i]) < 1e-10))
        s = qfunction(x)
        scores[i] = s

    scores[:large_window // 10] = 1.
    return scores, lw_mu, lw_std, sw_mu
