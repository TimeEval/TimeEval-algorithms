import numpy as np


def metric_matrix(img_shape, sigma=1.):
    M, N = img_shape

    X = np.arange(M)  # M,N
    Y = np.arange(N)

    P = (X[None, :, None, None] - X[None, None, None, :]) ** 2 \
        + (Y[:, None, None, None] - Y[None, None, :, None]) ** 2

    G = 1 / (2 * np.pi * sigma ** 2) * np.exp(- P / (2 * sigma ** 2))

    return G.reshape((M * N, M * N))


def imed_metric(a_imgs, b_imgs, G=None):
    assert a_imgs.shape == b_imgs.shape
    a_seq = a_imgs.reshape([a_imgs.shape[0], -1])
    b_seq = b_imgs.reshape([b_imgs.shape[0], -1])
    if G is None:
        G = metric_matrix(a_imgs.shape[1:])
    return np.array([(x - y).dot(G.dot(x - y)) for x, y in zip(a_seq, b_seq)])


def eucd_metric(a_imgs, b_imgs):
    assert a_imgs.shape == b_imgs.shape
    a_seq = a_imgs.reshape([a_imgs.shape[0], -1])
    b_seq = b_imgs.reshape([b_imgs.shape[0], -1])

    metric = (a_seq - b_seq) ** 2
    return metric.sum(axis=1)
