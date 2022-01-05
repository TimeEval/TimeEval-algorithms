import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
import numpy as np

from .preprocessor import Preprocessor
from .gat_layer import GraphAttentionLayer
from .forecasting import Forecasting
from .reconstruction import Reconstruction
from .dataset import TimeSeries
from .early_stopping import EarlyStopping


class MTAD_GAT(nn.Module):
    def __init__(self, num_features: int,
                 mag_window: int, score_window: int, batch_size: int, threshold: float, around_window_size: int,
                 kernel_size: int, window_size: int, gamma: float,
                 channel_sizes: List[int], latent_size: int, split: float, early_stopping_patience: int, early_stopping_delta: float):
        super().__init__()

        self.kernel_size = kernel_size
        self.window_size = window_size
        self.gamma = gamma

        self.preprocessor = Preprocessor(mag_window, score_window, batch_size, threshold, around_window_size)
        self.conv1d = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=self.kernel_size)
        self.feat_gat = GraphAttentionLayer(self.window_size - (self.kernel_size - 1))
        self.time_gat = GraphAttentionLayer(num_features)

        self.gru = nn.GRU(input_size=num_features * 3, hidden_size=channel_sizes[0])

        self.forecasting = Forecasting(channel_sizes, num_features)
        self.reconstruction = Reconstruction(channel_sizes, latent_size, num_features)

        self.split = split
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

    def loss(self, v_for: Tensor, v_for_target: Tensor, p_rec: Tensor, p_rec_target: Tensor, means: Tensor, stds: Tensor) -> Tensor:
        forecasting_loss = self.forecasting.loss(v_for, v_for_target)
        reconstruction_loss = self.reconstruction.loss.forward(p_rec, p_rec_target, means, stds)
        return forecasting_loss + reconstruction_loss

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        X = self.conv1d(X.transpose(2, 1)).transpose(2, 1)
        h_feat = self.feat_gat(X)
        h_time = self.time_gat(X.transpose(2, 1)).transpose(2, 1)
        X = torch.cat([h_feat, h_time, X], dim=2)
        X, _ = self.gru(X)

        v_for = self.forecasting(X[:, [-1], :])
        p_rec, means, stds = self.reconstruction(X[:, [-1], :])
        return v_for, p_rec, means, stds

    def _score(self, x: Tensor, x_hat: Tensor, p: Tensor) -> Tensor:
        scores = (torch.pow(x_hat - x, 2) + self.gamma * (1 - p)) / (1 + self.gamma)
        return torch.sum(scores, dim=2).reshape(-1)

    def fit(self, X: pd.DataFrame, epochs: int, lr: float, batch_size: int = 1, callback=None):
        self.train()

        X = self.preprocessor.forward(X)
        split_at = int(len(X)*self.split)
        dataset = TimeSeries(X[:split_at], self.window_size)
        valid_dataset = TimeSeries(X[split_at:], self.window_size)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

        early_stopping = EarlyStopping(self.early_stopping_patience, self.early_stopping_delta, epochs,
                                       callbacks=[(lambda i, _l, _e: callback(self) if i else None)])
        for epoch in early_stopping:
            self.train()
            print(f"Epoch {epoch+1}")
            for x, y in dataloader:
                optimizer.zero_grad()

                v_for, p_rec, mean, std = self.forward(x)
                loss = self.loss(v_for, y, p_rec, x[:, [0]], mean, std)
                loss.backward()
                optimizer.step()

            self.eval()
            val_losses = []
            for x, y in valid_dataloader:
                v_for, p_rec, mean, std = self.forward(x)
                loss = self.loss(v_for, y, p_rec, x[:, [0]], mean, std)
                val_losses.append(loss.item())
            val_losses = sum(val_losses) / len(val_losses)
            early_stopping.update(val_losses)

    def detect(self, X: pd.DataFrame, batch_size: int) -> np.ndarray:
        self.eval()

        X = self.preprocessor.forward(X)
        dataset = TimeSeries(X, self.window_size)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        scores = []
        for x, y in dataloader:
            v_for, p_rec, mean, std = self.forward(x)
            scores.append(self._score(y, v_for, p_rec))
        scores = torch.cat(scores).detach().numpy()
        return scores
