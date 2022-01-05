import torch
from torch.utils.data import Dataset


class TimeSeries(Dataset):
    def __init__(self, X, window_size: int):
        self.X = torch.from_numpy(X).float()
        self.window_size = window_size

    def __len__(self):
        return self.X.shape[0] - (self.window_size - 1)

    def __getitem__(self, index) -> torch.Tensor:
        end_idx = index+self.window_size
        x = self.X[index:end_idx].reshape(-1)
        return x
