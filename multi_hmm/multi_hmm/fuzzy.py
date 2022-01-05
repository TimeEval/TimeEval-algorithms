"""
Inspired by https://github.com/Fuminides/Fancy_aggregations/blob/master/Fancy_aggregations/integrals.py
"""


from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional
import numpy as np
import numpy.typing as npt
from fcmeans import FCM


class BaseFuzzyIntegrator(BaseEstimator, TransformerMixin):
    def __init__(self, measure: Optional[npt.ArrayLike] = None, axis: int = 0, p: int = 2):
        self.measure = measure
        self.axis = axis
        self.p = p

    def _generate_cardinality(self, n: int) -> npt.ArrayLike:
        card = np.arange(n, 0, -1)
        return (card / n) ** self.p

    def fit(self, X: npt.ArrayLike, y=None) -> 'BaseFuzzyIntegrator':
        if self.measure is None:
            self.measure = self._generate_cardinality(X.shape[self.axis])
            self.measure = np.expand_dims(self.measure, axis=1 - self.axis)
        return self

    def transform(self, X: npt.ArrayLike, y=None) -> npt.ArrayLike:
        assert self.measure is not None, f"Please fit {type(self).__name__} Transformer before transforming!"

        X_sorted = np.sort(X, axis=self.axis)
        return X_sorted


class Sugeno(BaseFuzzyIntegrator):
    def transform(self, X: npt.ArrayLike, y=None) -> npt.ArrayLike:
        X_sorted = super().transform(X, y)

        return np.amax(
            np.minimum(
                np.take(X_sorted, np.arange(0, X_sorted.shape[self.axis]), self.axis),
                self.measure
            ),
            axis=self.axis,
            keepdims=True
        )


class Choquet(BaseFuzzyIntegrator):
    def transform(self, X: npt.ArrayLike, y=None) -> npt.ArrayLike:
        X_sorted = super().transform(X, y)

        X_differenced = np.concatenate([
            np.take(X_sorted, [0], self.axis),
            np.diff(X_sorted, axis=self.axis)
        ], axis=self.axis)
        X_agg = np.dot(X_differenced.transpose((1 - self.axis, self.axis)), self.measure.reshape(-1))

        return X_agg.reshape(-1, 1)


class WrappedFCM(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins: int):
        self.fcm = FCM(n_clusters=n_bins)

    @property
    def n_bins(self) -> int:
        return self.fcm.n_clusters

    def fit(self, X: npt.ArrayLike, y=None) -> 'WrappedFCM':
        self.fcm.fit(X)
        return self

    def transform(self, X: npt.ArrayLike, y=None) -> npt.ArrayLike:
        return self.fcm.predict(X).reshape(-1, 1)

    def fit_transform(self, X, y=None, **fit_params) -> npt.ArrayLike:
        self.fit(X)
        return self.transform(X)
