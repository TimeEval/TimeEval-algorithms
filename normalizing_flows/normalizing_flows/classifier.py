import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Classifier(nn.Module):
    def __init__(self, n_features: int, layers: List[int], window_size: int, batch_size: int):
        super().__init__()

        layers = [n_features] + layers

        self.hidden_layers = nn.ModuleList([
            nn.Linear(layers[l], layers[l+1])
            for l in range(len(layers)-1)
        ])

        self.output_layer = nn.Linear(layers[-1], 2)
        self.window_size = window_size
        self.batch_size = batch_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            X = torch.relu(layer(X))
        X = torch.log_softmax(self.output_layer(X), dim=1)
        return X

    def detect(self, ts: np.ndarray) -> np.ndarray:
        self.eval()
        X = torch.from_numpy(sliding_window_view(ts, self.window_size, axis=0, writeable=True)).float()
        X = X.view(X.shape[0], -1)
        dataloader = DataLoader(TensorDataset(X), batch_size=self.batch_size)
        results = []
        for x in dataloader:
            output = self.forward(x[0])
            results.append(output[:, 1])
        results = torch.exp(torch.cat(results))
        return results.detach().numpy()
