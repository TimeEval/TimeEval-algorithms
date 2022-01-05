from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cluster import KMeans
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class KMeansAD(BaseEstimator, OutlierMixin):
    def __init__(self, k: int, window_size: int, stride: int, n_jobs: int):
        self.k = k
        self.window_size = window_size
        self.stride = stride
        self.model = KMeans(n_clusters=k, n_jobs=n_jobs)

    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        flat_shape = (X.shape[0] - (self.window_size - 1), -1)  # in case we have a multivariate TS
        slides = sliding_window_view(X, window_shape=self.window_size, axis=0).reshape(flat_shape)
        return slides[::self.stride, :]

    def fit(self, X: np.ndarray, y=None, preprocess=True) -> 'KMeansAD':
        if preprocess:
            X = self._preprocess_data(X)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray, preprocess=True) -> np.ndarray:
        if preprocess:
            X = self._preprocess_data(X)
        clusters = self.model.predict(X)
        diffs = np.linalg.norm(X - self.model.cluster_centers_[clusters], axis=1)
        return np.repeat(diffs, self.stride)

    def fit_predict(self, X, y=None) -> np.ndarray:
        X = self._preprocess_data(X)
        self.fit(X, y, preprocess=False)
        return self.predict(X, preprocess=False)
