import torch
from torch.utils.data import Dataset
from typing import Tuple


class TimeSeries(Dataset):
    def __init__(self, X, window_size: int):
        self.X = torch.from_numpy(X).float()
        self.window_size = window_size

    def __len__(self):
        return self.X.shape[0] - self.window_size

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        end_idx = index+self.window_size
        x = self.X[index:end_idx].view(-1)
        y = self.X[end_idx]
        return x, y
