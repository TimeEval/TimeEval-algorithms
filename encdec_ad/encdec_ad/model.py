import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Tuple
import os
import logging

from .dataset import TimeSeries
from .early_stopping import EarlyStopping


class Encoder(nn.Module):
    def __init__(self, input_size: int, lstm_layers: int, hidden_size: int):
        super().__init__()

        self.lstms = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, hidden = self.lstms(x)
        return hidden


class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, lstm_layers: int, prediction_length: int):
        super().__init__()

        self.prediction_length = prediction_length
        self.lstms = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.dense = nn.Linear(in_features=hidden_size, out_features=input_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        y = self.dense(hidden[0][[-1]].permute(1, 0, 2))
        y, hidden = self.lstms(y, hx=hidden)
        y = self.dense(y)
        outputs = [y]
        for i in range(1, self.prediction_length):
            if self.training:
                y, hidden = self.lstms(x[:, [-i]], hx=hidden)
            else:
                y, hidden = self.lstms(y, hx=hidden)
            y = self.dense(y)
            outputs = [y] + outputs
        return torch.cat(outputs, dim=1)


class AnomalyScorer:
    def __init__(self):
        super().__init__()

        self.mean = torch.tensor(0, dtype=torch.float64)
        self.var = torch.tensor(1, dtype=torch.float64)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        mean_diff = errors - self.mean
        return torch.mul(torch.mul(mean_diff, self.var**-1), mean_diff).mean(dim=1)

    def find_distribution(self, errors: torch.Tensor):
        self.mean = errors.mean(dim=[0, 1])
        self.var = errors.var(dim=[0, 1])

    def __str__(self) -> str:
        return f"AnomalyScorer(mean={self.mean},var={self.var})"


class EncDecAD(nn.Module):
    def __init__(self,
                 input_size: int,
                 latent_size: int,
                 lstm_layers: int,
                 split: float,
                 anomaly_window_size: int,
                 batch_size: int,
                 validation_batch_size: int,
                 test_batch_size: int,
                 epochs: int,
                 early_stopping_delta: float,
                 early_stopping_patience: int,
                 learning_rate: float,
                 *args, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.encoding_size = latent_size
        self.lstm_layers = lstm_layers
        self.split = split
        self.window_length = anomaly_window_size
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.lr = learning_rate

        self.encoder = Encoder(input_size=input_size, lstm_layers=lstm_layers, hidden_size=latent_size)
        self.decoder = Decoder(input_size=input_size, hidden_size=latent_size, lstm_layers=lstm_layers, prediction_length=anomaly_window_size)
        self.anomaly_scorer = AnomalyScorer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        x = self.decoder(x, hidden)
        return x

    def fit(self, ts: np.ndarray, model_path: os.PathLike, verbose=True):
        self.train()
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("EncDec-AD")
        optimizer = Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_dl, valid_dl = self._split_data(ts)
        def cb(i, _l, _e):
            if i:
                self._estimate_normal_distribution(valid_dl)
                self.save(model_path)
        early_stopping = EarlyStopping(self.early_stopping_patience, self.early_stopping_delta,
                                       self.epochs,
                                       callbacks=[cb])

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
        self._estimate_normal_distribution(valid_dl)

    def _estimate_normal_distribution(self, dl: DataLoader):
        self.eval()
        errors = []
        for x in dl:
            y_hat = self.forward(x)
            e = torch.abs(x - y_hat)
            errors.append(e)
        self.anomaly_scorer.find_distribution(torch.cat(errors))

    def _predict(self, x, criterion) -> torch.Tensor:
        y_hat = self.forward(x)
        loss = criterion(y_hat, x)
        return loss

    def anomaly_detection(self, ts: np.ndarray) -> np.ndarray:
        self.eval()
        dataloader = DataLoader(TimeSeries(ts, window_length=self.window_length),
                                batch_size=self.test_batch_size)
        errors = []
        for x in dataloader:
            y_hat = self.forward(x)
            e = torch.abs(x - y_hat)
            errors.append(e)
        errors = torch.cat(errors)
        return self.anomaly_scorer.forward(errors.mean(dim=1)).detach().numpy()

    def _split_data(self, ts: np.array) -> Tuple[DataLoader, DataLoader]:
        split_at = int(len(ts) * self.split)
        train_ts = ts[:split_at]
        valid_ts = ts[split_at:]
        train_ds = TimeSeries(train_ts, window_length=self.window_length)
        valid_ds = TimeSeries(valid_ts, window_length=self.window_length)
        return DataLoader(train_ds, batch_size=self.batch_size), DataLoader(valid_ds, batch_size=self.validation_batch_size)

    def save(self, path: os.PathLike):
        torch.save({
            "model": self.state_dict(),
            "anomaly_scorer": self.anomaly_scorer
        }, path)

    @staticmethod
    def load(path: os.PathLike, **kwargs) -> 'EncDecAD':
        checkpoint = torch.load(path)
        model = EncDecAD(**kwargs)
        model.load_state_dict(checkpoint["model"])
        model.anomaly_scorer = checkpoint["anomaly_scorer"]
        return model
