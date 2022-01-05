import torch

from torch.utils.data import Dataset
from typing import List, Optional, Union, Tuple


class TimeSeries(Dataset):
    def __init__(self, X, window_length: int, prediction_length: int = 1, output_dims: Optional[List[int]] = None):
        self.output_dims = output_dims or list(range(X.shape[1]))
        self.X = X
        self.window_length = window_length
        self.prediction_length = prediction_length

    def __len__(self):
        return self.X.shape[0] - (self.window_length - 1) - self.prediction_length

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        end_idx = index+self.window_length
        x = self.X[index:end_idx]
        if self.prediction_length == 0:
            return x
        y = self.X[end_idx:end_idx+self.prediction_length, self.output_dims]
        return x, y
