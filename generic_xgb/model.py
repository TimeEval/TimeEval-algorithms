import warnings
import joblib

import numpy as np

from typing import Optional
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import paired_distances
from xgboost import XGBRegressor

from numpy.lib.stride_tricks import sliding_window_view


class SlidingWindowProcessor(BaseEstimator, TransformerMixin):

    def __init__(self, window_size: int, standardize: bool = False):
        self.window_size = window_size
        if standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> 'SlidingWindowProcessor':
        if self.scaler:
            self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        y is unused (exists for compatibility)
        """
        if self.scaler:
            print("Standardizing input data")
            X = self.scaler.transform(X)
        X = X.reshape(-1)
        # the last window would have no target to predict, e.g. for n=10: [[1, 2] -> 3, ..., [8, 9] -> 10, [9, 10] -> ?]
        # fix XGBoost error by creating a copy instead of a view
        # xgboost.core.XGBoostError: ../src/c_api/../data/array_interface.h:234: Check failed: valid: Invalid strides in array.  strides: (1,1), shape: (3500, 100)
        new_X = sliding_window_view(X, window_shape=(self.window_size))[:-1].copy()
        new_y = np.roll(X, -self.window_size)[:-self.window_size]
        return new_X, new_y

    def transform_y(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            print("Standardizing input data")
            X = self.scaler.transform(X)
        return np.roll(X, -self.window_size)[:-self.window_size]

    def inverse_transform_y(self, y: np.ndarray, skip_inverse_scaling: bool = False) -> np.ndarray:
        result = np.full(shape=self.window_size+len(y), fill_value=np.nan)
        result[-len(y):] = y
        if not skip_inverse_scaling and self.scaler:
            print("Reversing standardization for prediction")
            result = self.scaler.inverse_transform(result)
        return result


class RandomForestAnomalyDetector(BaseEstimator, RegressorMixin):
    def __init__(self,
            train_window_size: int = 50,
            standardize: bool = False,
            n_estimators: float = 100,
            learning_rate: float = 0.01,
            booster: str = "gbtree",
            tree_method: str = "auto",
            n_trees: int = 1,
            max_depth: Optional[int] = None,
            max_samples: Optional[float] = None,
            colsample_bytree: Optional[float] = None,
            colsample_bylevel: Optional[float] = None,
            colsample_bynode: Optional[float] = None,
            random_state: int = 42,
            verbose: int = 0,
            n_jobs: int = 1,
            ):
        self.preprocessor = SlidingWindowProcessor(train_window_size, standardize)
        self.clf = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            booster=booster,
            tree_method=tree_method,
            num_parallel_tree=n_trees,
            max_depth=max_depth,
            subsamples=max_samples,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            random_state=random_state,
            verbosity=verbose,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'RandomForestAnomalyDetector':
        if y is not None:
            warnings.warn(f"y is calculated from X. Please don't pass y to RandomForestAnomalyDetector.fit, it will be ignored!")
        X, y = self.preprocessor.fit_transform(X)
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X, _ = self.preprocessor.transform(X)
        y_hat = self._predict_internal(X)
        return self.preprocessor.inverse_transform_y(y_hat)

    def detect(self, X: np.ndarray) -> np.ndarray:
        X, y = self.preprocessor.transform(X)
        y_hat = self._predict_internal(X)
        scores = paired_distances(y.reshape(-1, 1), y_hat.reshape(-1, 1)).reshape(-1)
        return self.preprocessor.inverse_transform_y(scores, skip_inverse_scaling=True)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> 'RandomForestAnomalyDetector':
        return joblib.load(path)
