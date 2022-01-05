import torch
from torch.utils.data import Dataset
import numpy as np

from img_embedding_cae.encoding import ScalogramEncoding


class TimeSeries(Dataset):

    def __init__(self, X: np.ndarray,
                       anomaly_window_size: int,
                       compression_const: int = 8):
        self.anomaly_window_size = anomaly_window_size
        self.img_size = anomaly_window_size // compression_const
        self.windows = self.create_windows(X)

    def create_windows(self, X):
        windows = [ScalogramEncoding.encode(X[i:i + self.anomaly_window_size], self.img_size)
                   for i in range(0, len(X) - self.anomaly_window_size, self.anomaly_window_size)]
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index) -> torch.Tensor:
        return self.windows[index]
