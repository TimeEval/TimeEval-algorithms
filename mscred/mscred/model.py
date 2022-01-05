from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Tuple, List
import numpy as np
import logging

from .convlstm import ConvLSTM, ConvLSTMAttention
from .correlation_matrix import CorrelationMatrix
from .early_stopping import EarlyStopping


def same_size_padding(w: int, k: int, s: int) -> int:
    return padding_from_size(w, w, k, s)


def padding_from_size(w_in: int, w_out: int, k: int, s: int) -> int:
    p = (w_out * s - s - w_in + k) / 2
    return int(np.ceil(p))


def output_size(w: int, k: int, s: int, p: int) -> int:
    return int(np.floor((w - k + 2*p) / s + 1))


class Encoder(nn.Module):
    def __init__(self, in_channels: int, image_dim: int):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=padding_from_size(image_dim, image_dim, 3, 1)),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=padding_from_size(image_dim, image_dim // 2, 2, 2)),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=padding_from_size(image_dim // 2, image_dim // 4, 2, 2)),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=padding_from_size(image_dim // 4, image_dim // 8, 2, 2))
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        xs: List[torch.Tensor] = [x]
        for conv in self.convs:
            x = conv(x)
            xs.append(x)
        return tuple(xs)


class Temporal(nn.Module):
    def __init__(self, image_dim: int):
        super().__init__()

        self.lstms = nn.ModuleList([
            ConvLSTM(32, 32, kernel_size=3, stride=1, padding=same_size_padding(image_dim, 3, 1)),
            ConvLSTM(64, 64, kernel_size=2, stride=2, padding=same_size_padding(image_dim // 2, 2, 2)),
            ConvLSTM(128, 128, kernel_size=2, stride=2, padding=same_size_padding(image_dim // 4, 2, 2)),
            ConvLSTM(256, 256, kernel_size=2, stride=2, padding=same_size_padding(image_dim // 8, 2, 2))
        ])

        self.attention = ConvLSTMAttention()

    def forward(self, *xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output: List[torch.Tensor] = []
        for x, lstm in zip(xs, self.lstms):
            lstm(x)
            x = self.attention(lstm.outputs)
            output.append(x)

        return tuple(output)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, image_dim: int):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=1, padding=padding_from_size(image_dim, image_dim, 3, 1))
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, padding=padding_from_size(image_dim, image_dim // 2, 2, 2))
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=padding_from_size(image_dim // 2, image_dim // 4, 2, 2))
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=padding_from_size(image_dim // 4, image_dim // 8, 2, 2))

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        x_hat = self.deconv4(xs[3])
        x_hat = self.deconv3(torch.cat([x_hat, xs[2]], dim=1))
        x_hat = self.deconv2(torch.cat([x_hat, xs[1]], dim=1))
        x_hat = self.deconv1(torch.cat([x_hat, xs[0]], dim=1))
        return x_hat


class MSCRED(nn.Module):
    def __init__(self, n_dimensions: int, windows: List[int], gap_time: int, window_size: int, batch_size: int, learning_rate: float, epochs: int,
                 early_stopping_patience: int, early_stopping_delta: float, split: float, test_batch_size: int):
        super().__init__()

        next_bigger_power = int(2 ** np.ceil(np.log2(n_dimensions)))
        self.n_dimensions = 8 if next_bigger_power < 8 else next_bigger_power

        self.windows = windows
        self.gap_time = gap_time
        self.window_size = window_size

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.split = split
        self.test_batch_size = test_batch_size

        self.encoder = Encoder(len(windows), self.n_dimensions)
        self.temporal = Temporal(self.n_dimensions)
        self.decoder = Decoder(len(windows), self.n_dimensions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = []
        for t in range(x.shape[1]):
            xs.append([x_i.unsqueeze(1) for x_i in self.encoder(x[:, t])])
        xs = [torch.cat(x_i, dim=1) for x_i in zip(*xs)]
        xs = xs[1:] if len(xs) > 1 else xs

        xs = self.temporal(*xs)
        x = self.decoder(*xs)

        return x

    def fit(self, data: np.ndarray, args):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("MSCRED")

        split_at = int(len(data) * self.split)
        dataset = CorrelationMatrix(data[:split_at], self.windows, self.gap_time, self.window_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        valid_dataset = CorrelationMatrix(data[split_at:], self.windows, self.gap_time, self.window_size)
        valid_loader = DataLoader(valid_dataset, batch_size=self.test_batch_size)
        criterion = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        early_stopping = EarlyStopping(patience=self.early_stopping_patience, delta=self.early_stopping_delta, epochs=self.epochs,
                                       callbacks=[(lambda i, _l, _e: self.save(args) if i else None)])

        for e in early_stopping:
            self.train()
            losses = []
            for x in dataloader:
                self.zero_grad()
                x_hat = self.forward(x)
                loss = criterion(torch.flatten(x_hat, start_dim=1), torch.flatten(x[:, -1], start_dim=1))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self.eval()
            val_losses = []
            for x in valid_loader:
                x_hat = self.forward(x)
                loss = criterion(torch.flatten(x_hat, start_dim=1), torch.flatten(x[:, -1], start_dim=1))
                val_losses.append(loss.item())
            val_losses = sum(val_losses) / len(val_losses)
            early_stopping.update(val_losses)

            logger.info(f"Epoch {e} | Training Loss: {sum(losses) / len(losses):.3f} | Validation Loss: {val_losses:.3f}")

    def detect(self, data: np.ndarray) -> np.ndarray:
        self.eval()
        dataset = CorrelationMatrix(data, self.windows, self.gap_time, self.window_size)
        dataloader = DataLoader(dataset, batch_size=self.test_batch_size)
        losses = []
        for x in dataloader:
            x_hat = self.forward(x)
            loss = F.mse_loss(torch.flatten(x_hat, start_dim=1), torch.flatten(x[:, -1], start_dim=1), reduction="none")
            losses.append(loss)
        return torch.cat(losses, dim=0).mean(dim=1).detach().numpy()

    def save(self, args):
        hyper_params = asdict(args.customParameters)
        hyper_params["n_dimensions"] = self.n_dimensions
        del hyper_params["random_state"]
        with open(args.modelOutput, "wb") as f:
            torch.save({
                "state_dict": self.state_dict(),
                "hyper_params": hyper_params
            }, f)

    @staticmethod
    def load(args) -> 'MSCRED':
        with open(args.modelInput, "rb") as f:
            loaded = torch.load(f)
        model = MSCRED(**loaded["hyper_params"])
        model.load_state_dict(loaded["state_dict"])
        return model
