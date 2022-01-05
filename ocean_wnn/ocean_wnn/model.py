import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import logging
import os
from sklearn.preprocessing import StandardScaler

from .early_stopping import EarlyStopping
from .dataset import TimeSeries


class WBF(nn.Module):  # Wavelet Basis Function
    def __init__(self, a: float, k: float, mother_wavelet: str = "mexican_hat", C: float = 1.75):
        # a: Scale parameter, k: Shift parameter

        super().__init__()
        self.a = a
        self.k = k
        self.C = C

        self.mother_wavelet = mother_wavelet

    def _mexican_hat(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp((-x**2) / 2) * (1 - x**2)

    def _central_symmetric(self, x: torch.Tensor) -> torch.Tensor:
        return -x * torch.exp((-x**2) / 2)

    def _morlet(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp((-x**2) / 2) * torch.cos(self.C * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mother_wavelet == "mexican_hat":
            fn = self._mexican_hat
        elif self.mother_wavelet == "central_symmetric":
            fn = self._central_symmetric
        else: # if self.mother_wavelet == "morlet"
            fn = self._morlet

        return abs(self.a)**(-0.5) * fn((x - self.k) / self.a)


class WNN(nn.Module):
    def __init__(self, window_size: int, hidden_size: int, a: float, k: float, wbf: str, C: float):
        super().__init__()

        self.window_size = window_size

        self.fc0 = nn.Linear(window_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.wbf = WBF(a, k, mother_wavelet=wbf, C=C)
        self.scaler = StandardScaler()

        self.error_dist: Optional[torch.distributions.Normal] = None
        self.threshold: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wbf(self.fc0(x))
        x = self.fc1(x)
        return x

    def fit(self, X: np.ndarray, epochs: int, learning_rate: float,
            batch_size: int, test_batch_size: int, split: float,
            early_stopping_delta: float, early_stopping_patience: int, threshold_percentile: float,
            verbose: bool = True, model_path: os.PathLike = "./model.th") -> 'WNN':
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("WNN")

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        X = self.scaler.fit_transform(X)
        train_dl, valid_dl = self._split_data(X, batch_size, test_batch_size, split)
        early_stopping = EarlyStopping(delta=early_stopping_delta, patience=early_stopping_patience, epochs=epochs,
                                       callbacks=[(lambda i, _l, _e: self.save(model_path) if i else None)])

        for epoch in early_stopping:
            self.train()
            train_losses = []
            for x, y in train_dl:
                optimizer.zero_grad()
                x_hat = self.forward(x)
                loss = criterion(x_hat, y)
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()

            self.eval()
            losses = []
            for x, y in valid_dl:
                x_hat = self.forward(x)
                loss = criterion(x_hat, y)
                losses.append(loss.item())
            early_stopping.update(sum(losses)/len(losses))
            if verbose:
                logger.info(
                    f"Epoch {epoch}: Training Loss {sum(train_losses) / len(train_dl)} \t "
                    f"Validation Loss {sum(losses) / len(valid_dl)}"
                )
        self._calculate_residual_error_distribution(valid_dl, threshold_percentile)
        return self

    def _calculate_residual_error_distribution(self, dataloader: DataLoader, p_u: float):
        self.eval()
        losses = []
        for x, y in dataloader:
            x_hat = self.forward(x)
            loss = F.mse_loss(x_hat, y, reduction="none")
            losses.append(loss)
        losses = torch.cat(losses)
        self.error_dist = torch.distributions.Normal(loc=torch.mean(losses), scale=torch.std(losses))
        p_d = 1 - p_u
        self.threshold = 0.5 * (abs(self.error_dist.loc + torch.log(torch.tensor(p_u / p_d)) * self.error_dist.scale)
                                + abs(self.error_dist.loc + torch.log(torch.tensor(p_d / p_u)) * self.error_dist.scale))
        print("threshold", self.threshold)

    @torch.no_grad()
    def detect(self, X: np.ndarray, with_threshold: bool = True):
        self.eval()
        X = self.scaler.transform(X)
        dataset = TimeSeries(X, window_size=self.window_size)

        losses = [torch.tensor([np.nan])] * self.window_size
        predictions = []
        next_window, y = dataset[0]
        exceeding_points = []
        for i in range(1, len(dataset) + 1):
            x_hat = self.forward(next_window.view(1, -1))
            loss = F.mse_loss(x_hat, y, reduction="none")
            losses.append(loss.view(-1))
            predictions.append(x_hat.item())
            if i < len(dataset):
                next_window, y = dataset[i]
                if with_threshold:
                    if loss > self.threshold:
                        exceeding_points.append(i)
                        next_window[-1] = x_hat.view(-1)
        return torch.flatten(torch.cat(losses)).detach().numpy(), exceeding_points

    def _split_data(self, ts: np.array, batch_size: int, test_batch_size: int, split: float) -> Tuple[DataLoader, DataLoader]:
        split_at = int(len(ts) * split)
        train_ts = ts[:split_at]
        valid_ts = ts[split_at:]
        train_ds = TimeSeries(train_ts, window_size=self.window_size)
        valid_ds = TimeSeries(valid_ts, window_size=self.window_size)
        return DataLoader(train_ds, batch_size=batch_size), DataLoader(valid_ds, batch_size=test_batch_size)

    def save(self, path: os.PathLike):
        torch.save(self, path)

    @staticmethod
    def load(path: os.PathLike) -> 'WNN':
        model = torch.load(path)
        return model
