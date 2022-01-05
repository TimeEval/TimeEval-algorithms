from sklearn.base import BaseEstimator, OutlierMixin
from typing import List
import numpy as np
import torch
import os

from .dae import DAE, Activation
from .knn import KNNEnsemble


class HybridKNN(BaseEstimator, OutlierMixin):
    def __init__(self, layer_sizes: List[int],
                 activation: Activation,
                 split: float,
                 anomaly_window_size: int,
                 batch_size: int,
                 test_batch_size: int,
                 epochs: int,
                 early_stopping_delta: float,
                 early_stopping_patience: int,
                 learning_rate: float,
                 n_neighbors: int,
                 n_estimators: int,
                 *args,
                 **kwargs):
        self.dae = DAE(layer_sizes, activation, split, anomaly_window_size, batch_size, test_batch_size, epochs,
                       early_stopping_delta, early_stopping_patience, learning_rate)
        self.knn = KNNEnsemble(n_neighbors, n_estimators)

    def fit(self, X: np.ndarray, model_path: os.PathLike) -> 'HybridKNN':
        def knn_callback(i, _l, _e):
            if i:
                self.knn.fit(self.dae.reduce(X))

        def save_callback(i, _l, _e):
            if i:
                self.save(model_path)

        self.dae.fit(X, callbacks=[knn_callback, save_callback])
        reduced_X = self.dae.reduce(X)
        self.knn.fit(reduced_X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        reduced_X = self.dae.reduce(X)
        scores = self.knn.predict(reduced_X)
        return scores

    def save(self, path: os.PathLike):
        torch.save(self, path)

    @staticmethod
    def load(path: os.PathLike) -> 'HybridKNN':
        model = torch.load(path)
        return model
