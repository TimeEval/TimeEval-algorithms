import torch
from torch.utils.data import Dataset


class TimeSeries(Dataset):
    def __init__(self, X, window_length: int):
        self.X = torch.from_numpy(X).float()
        self.window_length = window_length

    def __len__(self):
        return self.X.shape[0] - (self.window_length * 2 - 1)

    def __getitem__(self, index) -> torch.Tensor:
        end_idx = index+self.window_length
        x = self.X[index:end_idx]
        return x
