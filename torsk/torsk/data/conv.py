import numpy as np
from scipy.signal import convolve2d


def _mean_kernel(kernel_shape):
    return np.ones(kernel_shape) / (kernel_shape[0] * kernel_shape[1])


def _random_kernel(kernel_shape):
    return np.random.uniform(size=kernel_shape, low=-1, high=1)


def _gauss_kernel(kernel_shape):
    ysize, xsize = kernel_shape
    yy = np.linspace(-ysize / 2., ysize / 2., ysize)
    xx = np.linspace(-xsize / 2., xsize / 2., xsize)
    sigma = min(kernel_shape) / 6.

    gaussian = np.exp(-(xx[:, None]**2 + yy[None, :]**2) / (2 * sigma**2))
    norm = np.sum(gaussian)  # L1-norm is 1
    gaussian = (1. / norm) * gaussian

    return gaussian


def get_kernel(kernel_shape, kernel_type, dtype):
    if kernel_type == "mean":
        kernel = _mean_kernel(kernel_shape)
    elif kernel_type == "random":
        kernel = _random_kernel(kernel_shape)
    elif kernel_type == "gauss":
        kernel = _gauss_kernel(kernel_shape)
    else:
        raise NotImplementedError(f"Unkown kernel type `{kernel_type}`")
    return kernel.astype(dtype)


def _conv_out_size(in_size, kernel_size, padding=0, dilation=1, stride=1):
    num = in_size + 2 * padding - dilation * (kernel_size - 1) - 1
    size = num // stride + 1
    return size


def conv2d_output_shape(in_size, size, padding=0, dilation=1, stride=1):
    """Calculate output shape of a convolution of an image of in_shape.
    Formula taken form pytorch conv2d
    """
    height = _conv_out_size(in_size[0], size[0], padding, dilation, stride)
    width = _conv_out_size(in_size[1], size[1], padding, dilation, stride)
    return (height, width)


def conv2d(image, kernel_type, size):
    kernel = get_kernel(size, kernel_type)
    conv = convolve2d(image, kernel, mode="valid")
    return conv


def conv2d_sequence(sequence, kernel_type, size):
    """2D convolution of a sequence of images. Convolution mode is valid, which
    means that only values which do not need to be padded are calculated.

    Params
    ------
    sequence : ndarray
        with shape (time, ydim, xdim)
    kernel_type : str
        one of `gauss`, `mean`, `random`
    size : tuple
        shape of created convolution kernel

    Returns
    -------
    ndarray
        convoluted squence of shape (time, height, width). Height and width
        can be calculated with conv2d_output_shape
    """
    kernel = get_kernel(size, kernel_type)
    conv = [convolve2d(img, kernel, mode="valid") for img in sequence]
    return np.asarray(conv, dtype=sequence.dtype)
