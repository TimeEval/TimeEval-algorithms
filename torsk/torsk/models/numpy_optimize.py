import logging
import numpy as np

from torsk.data.utils import svd, lstsq
from torsk.numpy_accelerate import to_bh, to_np, bh_dot

logger = logging.getLogger(__name__)

def _extended_states(inputs, states):
    ones = np.ones([inputs.shape[0], 1], dtype=inputs.dtype)
    X    = np.concatenate([ones, to_np(inputs), to_np(states)], axis=1).T
    return X


def _pseudo_inverse_svd(inputs, states, labels, timer=None):
    timer.begin("pseudo_inverse_svd")
    X = _extended_states(inputs, states)
    U, s, Vh = svd(X,timer)
    scale = s[0]
    n = len(s[np.abs(s / scale) > 1e-4])  # Ensure condition number less than 10.000

    U, s, Vh = to_bh(U), to_bh(s), to_bh(Vh)
    L = labels.T

    v = Vh[:n, :].T
    uh = U[:, :n].T

    wout = bh_dot(bh_dot(L, v) / s[:n], uh)

    timer.end()
    return wout


def _pseudo_inverse_lstsq(inputs, states, labels, timer=None):
    timer.begin("pseudo_inverse_lstsq")
    X = _extended_states(inputs, states)

    wout, _, _, s = lstsq(X.T, labels,timer)
    condition = s[0] / s[-1]

    wout, s =  to_bh(wout), to_bh(s)

    if(np.log2(np.abs(condition)) > 12):  # More than half of the bits in the data are lost
        logger.warning(
            f"Large condition number in pseudoinverse: {condition}"
            " losing more than half of the digits. Expect numerical blowup!")
        logger.warning(f"Largest and smallest singular values: {s[0]}  {s[-1]}")

    timer.end()
    return wout.T


def pseudo_inverse(inputs, states, labels, mode="svd", timer=None):
    if mode == "svd":
        return _pseudo_inverse_svd(inputs, states, labels, timer)
    elif mode == "lstsq":
        return _pseudo_inverse_lstsq(inputs, states, labels, timer)
    else:
        raise ValueError(f"Unknown mode: '{mode}'")


def tikhonov(inputs, states, labels, beta):
    X = to_np(_extended_states(inputs, states))

    Id = np.eye(X.shape[0])
    A = np.dot(X, X.T) + beta + Id
    B = np.dot(X, labels)

    # Solve linear system instead of calculating inverse
    wout = np.linalg.solve(A, B)
    return to_bh(wout.T)
