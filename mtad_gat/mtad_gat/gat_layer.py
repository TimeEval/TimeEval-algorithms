import torch
from torch import Tensor
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    def __init__(self, len_ts: int):
        super().__init__()

        self.w = nn.Linear(2*len_ts, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU()

    def alpha(self, X: Tensor) -> Tensor:
        e = torch.zeros(X.shape[0], X.shape[2], X.shape[2])
        for i in range(X.shape[2]):
            for j in range(X.shape[2]):
                if i != j:
                    e[:, i, j] = self.leaky_relu(self.w(torch.cat([X[:, :, i], X[:, :, j]], dim=1))).reshape(-1)
        alpha = torch.softmax(e, dim=1)
        return alpha

    def forward(self, X: Tensor) -> Tensor:
        alpha = self.alpha(X)
        h = torch.bmm(alpha, X.transpose(2, 1)).transpose(2, 1)
        return torch.sigmoid(h)
