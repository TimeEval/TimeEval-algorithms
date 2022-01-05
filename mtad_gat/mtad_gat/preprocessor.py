import torch
from torch import Tensor
import numpy as np
import pandas as pd
from typing import Optional
from msanomalydetector.spectral_residual import SpectralResidual


class Preprocessor:
    def __init__(self, mag_window: int, score_window: int, batch_size: int, threshold: float, around_window_size: int):
        self.training = True
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None

        self.mag_window = mag_window
        self.score_window = score_window
        self.batch_size = batch_size
        self.threshold = threshold

        self.around_window_size = around_window_size

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _normalization(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.training:
            self.min = np.min(X.iloc[:, 1:].values, axis=0)
            self.max = np.max(X.iloc[:, 1:].values, axis=0)
        for c in range(1, X.shape[1]):
            X.iloc[:, c] = (X.iloc[:, c] - self.min[c-1]) / (self.max[c-1] - self.min[c-1])
        return X

    def _window_mean_anomalies(self, sr: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        result = X.iloc[:, 1:].values
        shape = result.shape
        sr = sr.flatten()
        indices = np.where(sr > self.threshold)[0]

        for idx in indices:
            window = np.arange(max(0, idx - self.around_window_size),
                               min(sr.shape[0], idx + self.around_window_size + 1))
            window = window[~np.isin(window, indices)]
            result[idx] = result[window].mean()

        return result.reshape(shape)

    def _cleaning(self, X: pd.DataFrame) -> np.ndarray:
        sr = np.zeros((X.shape[0], X.shape[1]-1))

        for c in range(1, X.shape[1]):
            series = X.iloc[:, [0, c]]
            series.columns = ["timestamp", "value"]
            sr[:, c-1] = SpectralResidual(
                series,
                mag_window=self.mag_window,
                score_window=self.score_window,
                batch_size=self.batch_size,
                sensitivity=99,
                detect_mode='AnomalyOnly',
                threshold=self.threshold
            ).detect().score.values

        result = self._window_mean_anomalies(sr, X)
        return result

    def forward(self, X: pd.DataFrame) -> Tensor:
        with torch.no_grad():
            X = self._normalization(X)
            if self.training:
                X = self._cleaning(X)
            return torch.from_numpy(X).float()
