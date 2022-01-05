import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List
import logging


class CorrelationMatrix(Dataset):
    def __init__(self, data: np.ndarray, windows: List[int], gap_time: int, window_size: int):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CorrelationMatrix")
        self.windows = windows
        self.gap_time = gap_time
        self.window_size = window_size
        self.X = self._transform(torch.from_numpy(data))

    def _add_padding(self, cm: torch.Tensor) -> torch.Tensor:
        n_dim = cm.shape[2]
        next_bigger_power = int(2**np.ceil(np.log2(n_dim)))
        next_bigger_power = 8 if next_bigger_power < 8 else next_bigger_power
        padded = torch.zeros(cm.shape[0], cm.shape[1], next_bigger_power, next_bigger_power)
        padded[:, :, :n_dim, :n_dim] = cm
        return padded

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        self.logger.info("Generating correlation matrices")
        min_val = X.min(dim=1).values[:, None]
        X = (X - min_val) / (X.max(dim=1).values[:, None] - min_val + 1e-6)

        length = len(X)
        windowed_length = X.shape[0] - (max(self.windows) - 1)
        dimensions = X.shape[1]
        n_matrices = int(np.ceil(windowed_length / self.gap_time))
        offset = 0 if (windowed_length % self.gap_time == 0) else self.gap_time
        correlation_matrix = torch.zeros(n_matrices + (1 if offset > 0 else 0), len(self.windows), dimensions, dimensions)
        for w, win in enumerate(self.windows):
            for t_, t in enumerate(range(max(self.windows), length + offset, self.gap_time)):
                for i in range(dimensions):
                    for j in range(dimensions):
                        correlation_matrix[t_, w, i, j] = torch.dot(X[t - win:t, i], X[t - win:t, j]) / win
        return self._add_padding(correlation_matrix)

    def __len__(self):
        return self.X.shape[0] - (self.window_size - 1)

    def __getitem__(self, item: int):
        return self.X[item:item+self.window_size]
