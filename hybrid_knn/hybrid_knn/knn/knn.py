import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple
import logging


class KNNEnsemble(BaseEstimator, OutlierMixin):
    def __init__(self, k: int, l: int):
        self.k = k
        self.l = l
        self.s: List[np.ndarray] = []
        self.g: List[np.ndarray] = []
        self.knn_models: List[KNN] = []

    def fit(self, X: np.ndarray) -> 'KNNEnsemble':
        self._eventually_reduce_estimator_size(X.shape)
        np.random.shuffle(X)
        self.s = np.array_split(X, self.l)
        if not self._validate_fitting():
            self._init_knn_models()
        for knn, s in zip(self.knn_models, self.s):
            knn.fit(s)
        self.g = [self._calculate_kth_distance(s) for s in self.s]

        return self

    def _init_knn_models(self):
        self.knn_models = [KNN(self.k) for _ in range(self.l)]

    def _calculate_kth_distance(self, X: np.ndarray) -> np.ndarray:
        if not self._validate_fitting():
            raise NotFittedError("This KNNEnsemble instance is not fitted yet. "
                                 "Call 'fit' with appropriate arguments before using this estimator.")

        d = np.zeros((X.shape[0], self.l))
        for l, knn in enumerate(self.knn_models):
            d[:, l] = knn.kth_neighbor_distance(X)
        g = d.mean(axis=1)
        return g

    def _validate_fitting(self) -> bool:
        return len(self.knn_models) > 0

    def _eventually_reduce_estimator_size(self, data_shape: Tuple[int, int]):
        max_estimators = data_shape[0] // self.k
        if max_estimators < self.l:
            logging.warning(f"The dataset is too small for the number of estimators ({self.l}). "
                            f"We set `n_estimators` to {max_estimators}!")
        self.l = min(max_estimators, self.l)

    def predict(self, X: np.ndarray) -> np.ndarray:
        g = self._calculate_kth_distance(X)
        p = np.zeros((X.shape[0], self.l))
        for l in range(self.l):
            identity = np.greater(g.reshape(-1, 1), self.g[l].reshape(1, -1)).astype(int)
            p[:, l] = identity.mean(axis=1)
        return p.mean(axis=1)


class KNN(BaseEstimator, OutlierMixin):
    def __init__(self, k: int):
        self.k = k
        self.nbrs = NearestNeighbors()

    def fit(self, X: np.ndarray) -> 'KNN':
        self.nbrs.fit(X)
        return self

    def kth_neighbor_distance(self, X: np.ndarray) -> float:
        distances, _ = self.nbrs.kneighbors(X, self.k, return_distance=True)
        return distances[:, -1]
