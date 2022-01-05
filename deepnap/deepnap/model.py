from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from copy import deepcopy
import logging
from typing import List, Tuple

from .dataset import TimeSeries
from .early_stopping import EarlyStopping


class JointReconstructionLoss(_Loss):
    def __init__(self, window_size: int, partial_sequence_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.window_size = window_size
        self.partial_sequence_length = partial_sequence_length

    def forward(self, input: torch.Tensor, target: torch.Tensor, partial_reconstructions: torch.Tensor):
        prediction_loss = F.mse_loss(input[:, self.window_size:], target, reduction=self.reduction)
        l = partial_reconstructions.shape[1]
        partial_reconstruction_losses = [F.mse_loss(partial_reconstructions[:, i], input[:, i:i+self.partial_sequence_length],
                                                      reduction=self.reduction) for i in range(l)]
        if self.reduction == "none":
            partial_reconstruction_loss = torch.cat(partial_reconstruction_losses, dim=1).sum(dim=1) / l
            prediction_loss = prediction_loss.mean(dim=[1,2])
            partial_reconstruction_loss = partial_reconstruction_loss.mean(dim=1)
        else:
            partial_reconstruction_loss = torch.stack(partial_reconstruction_losses).sum() / l
        return prediction_loss + partial_reconstruction_loss


class Prediction(nn.Module):
    def __init__(self, input_size: int, num_layers: int, dropout: float, hidden_size: int, prediction_length: int):
        super().__init__()

        self.prediction_length = prediction_length

        lstm_layers = []
        for l in range(num_layers):
            in_ = hidden_size
            if l == 0:
                in_ = input_size
            if l < num_layers - 1:
                out = hidden_size
            else:
                out = input_size

            lstm_layers.append(nn.LSTM(input_size=in_, hidden_size=out, batch_first=True))
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(lstm_layers)
        self.decoder = deepcopy(self.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode ts with len(self.encoder) lstm layers
        hidden_layers = torch.jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
        for encoder_lstm in self.encoder:
            x, hidden = encoder_lstm(x)
            x = self.dropout(x)
            hidden_layers.append(hidden)

        # only take last output
        x = x[:, -1].reshape(-1, 1, x.shape[2])

        predicted = []

        # predict the next self.prediction_length points of ts
        for k in range(self.prediction_length):
            # iterating through len(self.decoder) lstm layers
            for l, decoder_lstm in enumerate(self.decoder):
                x, hidden_layers[l] = decoder_lstm(x, hx=hidden_layers[l])
            predicted.append(x)
        return torch.cat(predicted, dim=1)


class Detection(nn.Module):
    def __init__(self, n_channels: int, window_size: int, partial_sequence_length: int, hidden_size: int):
        super().__init__()

        self.window_size = window_size
        self.partial_sequence_length = partial_sequence_length
        self.n_channels = n_channels

        input_size = 2 * window_size - partial_sequence_length
        self.linear_0 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear_1 = nn.Linear(in_features=hidden_size, out_features=partial_sequence_length)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        num_seq = self.window_size - (self.partial_sequence_length - 1)
        x_hat_primes = torch.zeros(x_hat.shape[0], num_seq, self.partial_sequence_length, x_hat.shape[2])

        # sliding window through predicted part of x_hat
        for i in range(num_seq):
            # extract partial sequence
            mask_start = self.window_size + i
            mask_end = self.window_size + i + self.partial_sequence_length
            x_hat_masked = torch.cat([x_hat[:, :mask_start], x_hat[:, mask_end:]], dim=1)
            # propagate through
            h = torch.relu(self.linear_0(x_hat_masked.reshape(x_hat_masked.shape[0], self.n_channels, -1)))
            x_hat_primes[:, i, :, :] = self.linear_1(h).reshape(x_hat.shape[0], -1, x_hat.shape[2])

        return x_hat_primes


class DeepNAP(nn.Module):
    def __init__(self, input_size: int, anomaly_window_size: int, partial_sequence_length: int,
                 lstm_layers: int, rnn_hidden_size: int, dropout: int, linear_hidden_size: int,
                 batch_size: int, epochs: int, learning_rate: float, early_stopping_delta: float, early_stopping_patience,
                 split: float, validation_batch_size: int, *args, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.window_size = anomaly_window_size
        self.prediction_length = anomaly_window_size
        self.partial_sequence_length = partial_sequence_length
        self.split = split
        self.validation_batch_size = validation_batch_size

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience

        self.prediction_module = torch.jit.script(Prediction(input_size=input_size,
                                                             num_layers=lstm_layers,
                                                             hidden_size=rnn_hidden_size,
                                                             prediction_length=self.prediction_length,
                                                             dropout=dropout))
        self.detection_module = torch.jit.script(Detection(n_channels=input_size,
                                                           window_size=self.window_size,
                                                           partial_sequence_length=self.partial_sequence_length,
                                                           hidden_size=linear_hidden_size))

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        predicted = self.prediction_module(x)
        # x_hat is both the original and the predicted values concatenated
        x_hat = torch.cat([x, predicted], dim=1)

        # x_hat_primes are smaller windows sequences within the predicted values
        x_hat_primes = self.detection_module(x_hat)
        return x_hat, x_hat_primes

    def fit(self, ts: np.ndarray, args, verbose=True):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("DeepNAP")

        split_at = int(len(ts) * self.split)
        dataset = TimeSeries(ts[:split_at], window_size=self.window_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        validation_dataset = TimeSeries(ts[split_at:], window_size=self.window_size)
        validation_loader = DataLoader(validation_dataset, batch_size=self.validation_batch_size)
        criterion = torch.jit.script(JointReconstructionLoss(window_size=self.window_size,
                                                             partial_sequence_length=self.partial_sequence_length))
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        early_stopping = EarlyStopping(self.early_stopping_patience, self.early_stopping_delta, self.epochs,
                                       callbacks=[(lambda i, _l, _e: self.save(args) if i else None)])
        for epoch in early_stopping:
            self.train()
            losses = []
            for x, y in dataloader:
                self.zero_grad()
                x_hat, x_hat_primes = self.forward(x)  # names from paper
                loss = criterion(x_hat, y, x_hat_primes)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self.eval()
            valid_losses = []
            for x, y in validation_loader:
                x_hat, x_hat_primes = self.forward(x)
                loss = criterion(x_hat, y, x_hat_primes)
                valid_losses.append(loss.item())
            validation_loss = sum(valid_losses)
            early_stopping.update(validation_loss)

            if verbose:
                logger.info(f"Epoch {epoch}: Avg Training Loss {sum(losses) / len(dataloader)}"
                            f"\tAvg Validation Loss {validation_loss / len(validation_loader)}")

    def anomaly_detection(self, ts: np.ndarray):
        self.eval()
        dataloader = DataLoader(TimeSeries(ts, window_size=self.window_size),
                                batch_size=self.validation_batch_size)
        criterion = torch.jit.script(JointReconstructionLoss(self.window_size, self.partial_sequence_length, reduction="none"))

        errors = []
        for x, y in dataloader:
            x_hat, x_hat_primes = self.forward(x)
            loss = criterion(x_hat, y, x_hat_primes)
            errors.append(loss)
        errors = torch.cat(errors)
        return errors.detach().numpy()

    def save(self, args):
        hyper_params = asdict(args.customParameters)
        hyper_params["input_size"] = self.input_size
        with open(args.modelOutput, "wb") as f:
            torch.save({
                "state_dict": self.state_dict(),
                "hyper_params": hyper_params
            }, f)

    @staticmethod
    def load(args) -> 'DeepNAP':
        with open(args.modelInput, "rb") as f:
            loaded = torch.load(f)
        model = DeepNAP(**loaded["hyper_params"])
        model.load_state_dict(loaded["state_dict"])
        return model
