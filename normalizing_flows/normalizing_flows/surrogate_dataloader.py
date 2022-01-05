import numpy as np
import torch
import torch.nn as nn
from nflows import distributions
from torch.utils.data import DataLoader
from scipy.stats import truncnorm
from scipy.stats import norm


class SurrogateDataLoader:
    def __init__(self, dataloader: DataLoader, student: nn.Module, distribution: distributions.StandardNormal, percentile: float = 0.05):
        self.dataloader = dataloader
        self.student = student

        self.dim = distribution._shape[0]
        self.loc = [0.] * self.dim
        self.scale = [1.] * self.dim

        self.bounds = [
            [
                norm.ppf(0.00, loc=self.loc, scale=self.scale),
                norm.ppf(percentile, loc=self.loc, scale=self.scale)
            ],
            [
                norm.ppf(1 - percentile, loc=self.loc, scale=self.scale),
                norm.ppf(1.00, loc=self.loc, scale=self.scale)
            ],
        ]

    def _sample_anomaly(self) -> torch.Tensor:
        result = [
            truncnorm.rvs(*self.bounds[0], size=(self.dataloader.batch_size // 2, self.dim), loc=self.loc, scale=self.scale),
            truncnorm.rvs(*self.bounds[1], size=(self.dataloader.batch_size // 2, self.dim), loc=self.loc, scale=self.scale),
        ]
        return torch.from_numpy(np.concatenate(result)).float()

    def _surrogate_anomaly(self) -> (torch.Tensor, torch.Tensor):
        samples = self._sample_anomaly()
        x, _ = self.student.forward(samples)
        return x, torch.ones(x.shape[0], dtype=torch.long)

    def __iter__(self):
        for x, y in self.dataloader:
            yield x, y
            yield self._surrogate_anomaly()
