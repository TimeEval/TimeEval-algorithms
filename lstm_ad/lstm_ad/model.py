import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer as BaseOptimizer
from torch.optim import Adam, SGD, RMSprop
from torch.utils.data import DataLoader
from typing import List, Tuple
from enum import Enum
import os
import logging

from .dataset import TimeSeries
from .early_stopping import EarlyStopping


class Optimizer(Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"

    def get(self, params, lr) -> BaseOptimizer:
        if self == Optimizer.ADAM:
            return Adam(params, lr=lr)
        elif self == Optimizer.SGD:
            return SGD(params, lr=lr)
        else:  # if self == Optimizer.RMSPROP:
            return RMSprop(params, lr=lr)


class AnomalyScorer:
    def __init__(self):
        super().__init__()

        self.mean = torch.tensor(0, dtype=torch.float64)
        self.var = torch.tensor(1, dtype=torch.float64)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        mean_diff = errors - self.mean
        return torch.mul(torch.mul(mean_diff, self.var**-1), mean_diff)

    def find_distribution(self, errors: torch.Tensor):
        self.mean = errors.mean(dim=[0, 1])
        self.var = errors.var(dim=[0, 1])


class LSTMAD(nn.Module):
    def __init__(self,
                 input_size: int,
                 lstm_layers: int,
                 split: float,
                 window_size: int,
                 prediction_window_size: int,
                 output_dims: List[int],
                 batch_size: int,
                 validation_batch_size: int,
                 test_batch_size: int,
                 epochs: int,
                 early_stopping_delta: float,
                 early_stopping_patience: int,
                 optimizer: str,
                 learning_rate: float,
                 *args, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.lstm_layers = lstm_layers
        self.split = split
        self.window_size = window_size
        self.prediction_length = prediction_window_size
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.optimizer = Optimizer(optimizer)
        self.lr = learning_rate
        if len(output_dims) > 0:
            self.hidden_units = len(output_dims)
        else:
            self.hidden_units = input_size

        self.lstms = nn.LSTM(input_size=input_size, hidden_size=self.hidden_units * self.prediction_length, batch_first=True, num_layers=lstm_layers)
        self.dense = nn.Linear(in_features=self.window_size * self.hidden_units * self.prediction_length, out_features=self.hidden_units * self.prediction_length)
        self.anomaly_scorer = AnomalyScorer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hidden = self.lstms(x)
        x = x.reshape(-1, self.window_size * self.hidden_units * self.prediction_length)
        x = self.dense(x)
        return x

    def fit(self, ts: np.ndarray, model_path: os.PathLike, verbose=True):
        self.train()
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("LSTM-AD")
        optimizer = self.optimizer.get(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_dl, valid_dl = self._split_data(ts)
        def cb(i, _l, _e):
            if i:
                self._estimate_normal_distribution(valid_dl)
                self.save(model_path)
        early_stopping = EarlyStopping(self.early_stopping_patience, self.early_stopping_delta, self.epochs,
                                       callbacks=[cb])

        for epoch in early_stopping:
            self.train()
            losses = []
            for x, y in train_dl:
                self.zero_grad()
                loss = self._predict(x, y, criterion)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self.eval()
            valid_losses = []
            for x, y in valid_dl:
                loss = self._predict(x, y, criterion)
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
        for x, y in dl:
            y_hat = self.forward(x)
            e = torch.abs(y.reshape(*y_hat.shape) - y_hat)
            errors.append(e)
        self.anomaly_scorer.find_distribution(torch.cat(errors))

    def _predict(self, x, y, criterion) -> torch.Tensor:
        y = y.reshape(-1, self.prediction_length * self.hidden_units)
        y_hat = self.forward(x)
        loss = criterion(y_hat, y)
        return loss

    def anomaly_detection(self, ts: np.ndarray) -> np.ndarray:
        self.eval()
        dataloader = DataLoader(TimeSeries(ts, window_length=self.window_size, prediction_length=self.prediction_length, output_dims=self.output_dims),
                                batch_size=self.test_batch_size)
        errors = []
        for x, y in dataloader:
            y_hat = self.forward(x)
            e = torch.abs(y.reshape(*y_hat.shape) - y_hat)
            errors.append(e)
        errors = torch.cat(errors)
        return self.anomaly_scorer.forward(errors.mean(dim=1)).detach().numpy()

    def _split_data(self, ts: np.array) -> Tuple[DataLoader, DataLoader]:
        split_at = int(len(ts) * self.split)
        train_ts = ts[:split_at]
        valid_ts = ts[split_at:]
        train_ds = TimeSeries(train_ts, window_length=self.window_size, prediction_length=self.prediction_length, output_dims=self.output_dims)
        valid_ds = TimeSeries(valid_ts, window_length=self.window_size, prediction_length=self.prediction_length, output_dims=self.output_dims)
        return DataLoader(train_ds, batch_size=self.batch_size), DataLoader(valid_ds, batch_size=self.validation_batch_size)

    def save(self, path: os.PathLike):
        torch.save({
            "model": self.state_dict(),
            "anomaly_scorer": self.anomaly_scorer
        }, path)

    @staticmethod
    def load(path: os.PathLike, **kwargs) -> 'LSTMAD':
        checkpoint = torch.load(path)
        model = LSTMAD(**kwargs)
        model.load_state_dict(checkpoint["model"])
        model.anomaly_scorer = checkpoint["anomaly_scorer"]
        return model
