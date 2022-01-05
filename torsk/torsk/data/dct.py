import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.linalg import lstsq


def sct_basis(nx, nk):
    """Basis for SCT (Slow Cosine Transform) which is a least-squares
    approximation to restricted DCT-III / Inverse DCT-II
    """
    xs = np.arange(nx)
    ks = np.arange(nk)
    basis = 2 * np.cos(np.pi * (xs[:, None] + 0.5) * ks[None, :] / nx)
    return basis


def sct(fx, basis):
    """SCT (Slow Cosine Transform) which is a least-squares approximation to
    restricted DCT-III / Inverse DCT-II
    """
    fk, _, _, _ = lstsq(basis, fx)
    return fk


def isct(fk, basis):
    """Inverse SCT"""
    fx = np.dot(basis, fk)
    return fx


def sct2(Fxx, basis1, basis2):
    """SCT of a two-dimensional array"""
    Fkx = sct(Fxx.T, basis2)
    Fkk = sct(Fkx.T, basis1)
    return Fkk


def isct2(Fkk, basis1, basis2):
    """Inverse SCT of a two-dimensional array"""
    Fkx = isct(Fkk.T, basis2)
    Fxx = isct(Fkx.T, basis1)
    return Fxx


def dct2(Fxx, nk1, nk2):
    """Two dimensional discrete cosine transform"""
    Fkk = dctn(Fxx, norm='ortho')[:nk1, :nk2]
    return Fkk


def idct2(Fkk, nx1, nx2):
    """Two dimensional inverse discrete cosine transform"""
    Fxx = idctn(Fkk, norm='ortho', shape=(nx1, nx2))
    return Fxx


def idct2_sequence(Ftkk, xsize):
    """Inverse Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftkk : ndarray with shape (time, nk1, nk2)
    size : (ny, nx) determines the resolution of the image

    Returns
    -------
    Ftxx: ndarray with shape (time, ny, nx)
    """
    Ftxx = idctn(Ftkk, norm='ortho', shape=xsize, axes=[1, 2])
    return Ftxx


def dct2_sequence(Ftxx, ksize):
    """Discrete Cosine Transform of a sequence of 2D images.

    Params
    ------
    Ftxx : ndarray with shape (time, ydim, xdim)
    size : (nk1, nk2) determines how many DCT coefficents are kept

    Returns
    -------
    Ftkk: ndarray with shape (time, nk1, nk2)
    """
    Ftkk = dctn(Ftxx, norm='ortho', axes=[1, 2])[:, :ksize[0], :ksize[1]]
    return Ftkk
