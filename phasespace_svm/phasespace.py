from typing import List, Sequence

import numpy as np
from sklearn.svm import OneClassSVM


def unfold(ts: np.ndarray, dim: int) -> np.ndarray:
    start = 0
    n = len(ts) - start - dim + 1
    index = start + np.expand_dims(np.arange(dim), 0) + np.expand_dims(np.arange(n), 0).T

    return ts[index]


def project(q: np.ndarray, dim: int) -> np.ndarray:
    ones = np.ones(dim)
    proj_matrix = np.identity(dim) - (1 / dim) * ones * ones.T
    return np.dot(q, proj_matrix)


def svm(X: np.ndarray, **svm_kwargs: dict) -> np.ndarray:
    clf = OneClassSVM(**svm_kwargs)
    clf.fit(X)
    scores = clf.decision_function(X)
    # invert decision_scores, outliers come with higher outlier scores
    return scores * -1


def align(x: np.ndarray, shape: Sequence[int], dim: int) -> np.ndarray:
    # vectors (windows) in phase space embedding are right aligned
    # --> fill start points with np.nan
    new_x = np.full(shape, np.nan)
    # new_x[dim-1:] = x
    # --> left alignment produces better results:
    new_x[:-dim+1] = x
    return new_x


def detect_anomalies(data: np.ndarray, embed_dims: List[int], projected_ps: bool = False,
                     **svm_kwargs: dict) -> np.ndarray:
    score_list = []
    for dim in embed_dims:
        Q = unfold(data, dim)
        if projected_ps:
            Q = project(Q, dim)
        scores = svm(Q, **svm_kwargs)
        scores = align(scores, shape=data.shape, dim=dim)
        score_list.append(scores)
    return np.nansum(np.array(score_list), axis=0)
