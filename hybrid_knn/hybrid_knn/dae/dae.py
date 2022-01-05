import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import List, Tuple, Callable
from enum import Enum
import logging

from .dataset import TimeSeries
from .early_stopping import EarlyStopping


class Activation(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"

    def __call__(self, *args, **kwargs) -> Callable:
        if self == Activation.RELU:
            return nn.ReLU()
        else:
            return nn.Sigmoid()


class Encoder(nn.Module):
    def __init__(self, layer_sizes: List[int], activation: Activation):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
        ])
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layer_sizes: List[int]):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i - 1]) for i in reversed(range(1, len(layer_sizes)))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DAE(nn.Module):
    def __init__(self,
                 layer_sizes: List[int],
                 activation: Activation,
                 split: float,
                 window_size: int,
                 batch_size: int,
                 test_batch_size: int,
                 epochs: int,
                 early_stopping_delta: float,
                 early_stopping_patience: int,
                 learning_rate: float):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.split = split
        self.window_size = window_size
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.lr = learning_rate

        self.encoder = Encoder(layer_sizes, self.activation)
        self.decoder = Decoder(layer_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, ts: np.ndarray, callbacks: List[Callable], verbose=True):
        self.train()
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("DAE")
        optimizer = Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_dl, valid_dl = self._split_data(ts)
        early_stopping = EarlyStopping(self.early_stopping_patience, self.early_stopping_delta, self.epochs, callbacks=callbacks)

        for epoch in early_stopping:
            self.train()
            losses = []
            for x in train_dl:
                self.zero_grad()
                loss = self._predict(x, criterion)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self.eval()
            valid_losses = []
            for x in valid_dl:
                loss = self._predict(x, criterion)
                valid_losses.append(loss.item())
            validation_loss = sum(valid_losses)
            early_stopping.update(validation_loss)
            if verbose:
                logger.info(
                    f"Epoch {epoch}: Training Loss {sum(losses) / len(train_dl)} \t "
                    f"Validation Loss {validation_loss / len(valid_dl)}"
                )

    def _predict(self, x, criterion) -> torch.Tensor:
        y_hat = self.forward(x)
        loss = criterion(y_hat, x)
        return loss

    def reduce(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        dl = DataLoader(TimeSeries(X, window_size=self.window_size), batch_size=self.test_batch_size)
        reduced_x: List[torch.Tensor] = []
        for x in dl:
            reduced_x.append(self.encoder(x))
        return torch.cat(reduced_x).detach().numpy()

    def _split_data(self, ts: np.array) -> Tuple[DataLoader, DataLoader]:
        split_at = int(len(ts) * self.split)
        train_ts = ts[:split_at]
        valid_ts = ts[split_at:]
        train_ds = TimeSeries(train_ts, window_size=self.window_size)
        valid_ds = TimeSeries(valid_ts, window_size=self.window_size)
        return DataLoader(train_ds, batch_size=self.batch_size), DataLoader(valid_ds, batch_size=self.test_batch_size)
