from typing import Tuple, List
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset import TimeSeries
from .early_stopping import EarlyStopping



class Encoder(nn.Module):

    def __init__(self, kernel_size: int, num_kernels: int, activation) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=num_kernels,
                      kernel_size=kernel_size,
                      stride=kernel_size),
            nn.Conv2d(in_channels=num_kernels,
                      out_channels=2*num_kernels,
                      kernel_size=kernel_size,
                      stride=kernel_size)
        ])
        self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class Decoder(nn.Module):

    def __init__(self, kernel_size: int, num_kernels: int, activation) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=2*num_kernels,
                               out_channels=num_kernels,
                               kernel_size=kernel_size,
                               stride=kernel_size),
            nn.ConvTranspose2d(in_channels=num_kernels,
                               out_channels=1,
                               kernel_size=kernel_size,
                               stride=kernel_size)
        ])
        self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class FullyConnected(nn.Module):

    def __init__(self, in_features: int,
                       conv_2_shape: torch.Size,
                       latent_size: int,
                       activation) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=latent_size),
            nn.Linear(in_features=latent_size, out_features=in_features),
        ])
        self.unflatten = nn.Unflatten(1, conv_2_shape)
        self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.unflatten(x)
        return x

# (anomaly_window_size//(8*kernel_size^2)+1)*(8*kernel_size^2)
class CAE(nn.Module):

    def __init__(self,
                 param_correction: bool,
                 downscaling_factor: int,
                 anomaly_window_size: int,
                 kernel_size: int,
                 num_kernels: int,
                 latent_size: int,
                 leaky_relu_alpha: float,
                 batch_size: int,
                 test_batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 split: float,
                 early_stopping_delta: float,
                 early_stopping_patience: int) -> None:
        
        super().__init__()
        
        if param_correction:
            if not anomaly_window_size % (downscaling_factor * kernel_size**2) == 0:
                anomaly_window_size = ((anomaly_window_size // (8*kernel_size**2)) + 1) * (8*kernel_size**2) 
        else:
            assert anomaly_window_size % (downscaling_factor * kernel_size**2) == 0

        self.downscaling_factor = downscaling_factor
        self.anomaly_window_size = anomaly_window_size
        self.latent_size = latent_size
        self.activation = nn.LeakyReLU(leaky_relu_alpha)
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.split = split
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience

        self.encoder = Encoder(kernel_size, num_kernels, self.activation)
        # [(img_size - kernel_size) / stride] + 1
        img_size = anomaly_window_size // downscaling_factor
        conv_1_size = ((img_size - kernel_size) // kernel_size) + 1
        conv_2_size = ((conv_1_size - kernel_size) // kernel_size) + 1
        conv_2_shape = torch.Size([2*num_kernels, conv_2_size, conv_2_size])
        out_features = conv_2_size**2 * (2*num_kernels)

        self.fully_connected = FullyConnected(out_features,
                                              conv_2_shape,
                                              self.latent_size,
                                              self.activation)

        self.decoder = Decoder(kernel_size, num_kernels, self.activation)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.fully_connected(x)
        x = self.decoder(x)
        return x


    def fit(self, ts: np.ndarray, model_path: os.PathLike, verbose=True) -> None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("CAE")
        optimizer = Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_dl, valid_dl = self.split_data(ts)
        early_stopping = EarlyStopping(self.early_stopping_patience, self.early_stopping_delta, self.epochs,
                                       callbacks=[(lambda i, _l, _e: self.save(model_path) if i else None)])

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


    def predict_ts(self, x, criterion) -> np.ndarray:
        self.eval()
        dl = DataLoader(TimeSeries(x, anomaly_window_size=self.anomaly_window_size, compression_const=self.downscaling_factor),
                        batch_size=1)
        scores: List[float] = []
        for x in dl:
            loss = self._predict(x, criterion)
            scores.append(loss.item())
        return np.array(scores)


    def split_data(self, ts: np.array) -> Tuple[DataLoader, DataLoader]:
        split_at = int(len(ts) * self.split)
        train_ts = ts[:split_at]
        valid_ts = ts[split_at:]
        train_ds = TimeSeries(train_ts, anomaly_window_size=self.anomaly_window_size, compression_const=self.downscaling_factor)
        valid_ds = TimeSeries(valid_ts, anomaly_window_size=self.anomaly_window_size, compression_const=self.downscaling_factor)
        return DataLoader(train_ds, batch_size=self.batch_size), DataLoader(valid_ds, batch_size=self.test_batch_size)


    def save(self, path: os.PathLike):
        torch.save(self, path)


    @staticmethod
    def load(path: os.PathLike) -> 'CAE':
        model = torch.load(path)
        return model
