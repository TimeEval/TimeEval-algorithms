import torch
from torch import Tensor
import torch.nn as nn
from typing import List


class Forecasting(nn.Module):
    def __init__(self, channel_sizes: List[int], num_features: int):
        super().__init__()

        channel_sizes.append(num_features)
        self.fcs = nn.ModuleList(
            nn.Linear(in_features=channel_sizes[c], out_features=channel_sizes[c + 1]) for c in
            range(len(channel_sizes) - 1)
        )

        self.loss = RMSELoss()

    def forward(self, X: Tensor) -> Tensor:
        for fc in self.fcs:
            X = torch.relu(fc(X))
        return X


class RMSELoss(nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.sqrt(super().forward(input, target))
        return loss
