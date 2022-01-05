import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeries(Dataset):
    def __init__(self, X: np.ndarray, window_size: int):
        self.X = torch.from_numpy(X).float()
        self.window_size = window_size

    def __len__(self):
        return self.X.shape[0] - (2 * self.window_size - 1)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        end_idx = index + self.window_size
        prediction_end_idx = end_idx + self.window_size
        x = self.X[index:end_idx]
        y = self.X[end_idx:prediction_end_idx]
        return x, y
