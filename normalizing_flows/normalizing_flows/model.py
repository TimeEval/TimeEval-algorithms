import torch
import torch.nn as nn
from nflows import transforms, distributions, flows
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import logging
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pathlib

from .classifier import Classifier
from .surrogate_dataloader import SurrogateDataLoader
from .early_stopping import EarlyStopping


class Teacher(nn.Module):
    def __init__(self, features: int, hidden_features: int, n_layers: int = 1):
        super().__init__()

        t = []
        for _ in range(n_layers):
            t.append(transforms.ReversePermutation(features=features))
            t.append(transforms.MaskedAffineAutoregressiveTransform(features=features, hidden_features=hidden_features))
        self.transform = transforms.CompositeTransform(t)

        self.domain_distribution = distributions.StandardNormal(shape=[features])
        self.flow = flows.Flow(transform=self.transform, distribution=self.domain_distribution)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z_0, _ = self.transform.inverse(X)
        return z_0

    def log_prob(self, z_0: torch.Tensor) -> torch.Tensor:
        return self.domain_distribution.log_prob(z_0)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.flow.sample(num_samples)


class Student(nn.Module):
    def __init__(self, teacher_kwargs: Dict):
        super().__init__()

        self.inverse_transform = transforms.InverseTransform(transforms.MaskedAffineAutoregressiveTransform(**teacher_kwargs))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z_n = self.inverse_transform.forward(X)
        return z_n


class NormalizingFlow(nn.Module):
    def __init__(self, n_features: int, n_hidden_features: int, hidden_layer_shape: List[int], window_size: int, split: float,
                 epochs: int, batch_size: int, test_batch_size: int, teacher_epochs: int, distillation_iterations: int,
                 percentile: float, early_stopping_patience: int, early_stopping_delta: float):
        super().__init__()

        teacher_kwargs = {"features": n_features, "hidden_features": n_hidden_features}
        self.teacher = Teacher(**teacher_kwargs)
        self.student = Student(teacher_kwargs)
        self.classifier = Classifier(n_features, hidden_layer_shape, window_size, test_batch_size)
        self.window_size = window_size
        self.split = split
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.teacher_epochs = teacher_epochs
        self.distillation_iterations = distillation_iterations
        self.distillation_batch_size = batch_size
        self.percentile = percentile
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("NormalizingFlow")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.teacher.forward(X)

    def fit_teacher(self, X: TensorDataset, epochs: int):
        dataloader = DataLoader(X, batch_size=self.batch_size)

        self.teacher.train()
        optimizer = Adam(self.teacher.parameters())
        for e in range(epochs):
            losses = []
            for x, y in dataloader:
                optimizer.zero_grad()
                z_0 = self.teacher.forward(x)
                loss = -self.teacher.log_prob(z_0).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            self.logger.info(f"training loss {sum(losses) / len(losses)}")

    def fit_student(self, iterations: int, batch_size: int):
        self.teacher.eval()
        loss_fn = nn.KLDivLoss()
        optimizer = Adam(self.student.parameters())
        for i in range(iterations):
            optimizer.zero_grad()
            Ps = self.teacher.sample(batch_size)
            z_n, _ = self.student.forward(Ps)
            Pt = self.teacher.forward(z_n)
            loss = loss_fn(Pt, Ps)
            loss.backward()
            optimizer.step()

    def fit(self, ts: np.ndarray, targets: np.ndarray, verbose=True, model_path: pathlib.Path = pathlib.Path("./model.th")) -> 'NormalizingFlow':

        X = torch.from_numpy(sliding_window_view(ts, self.window_size, axis=0)).float()
        X = X.view(len(X), -1)
        y = torch.from_numpy((sliding_window_view(targets, self.window_size, axis=0) == 1).any(axis=1)).long()

        dataset = TensorDataset(X, y)

        self.logger.info("----- Training Teacher -----")
        self.fit_teacher(dataset, self.teacher_epochs)

        self.logger.info("----- Training Student -----")
        self.fit_student(self.distillation_iterations, self.distillation_batch_size)

        self.teacher.eval()
        self.student.eval()

        self.logger.info("----- Training Classifier -----")

        optimizer = Adam(self.classifier.parameters())
        criterion = nn.NLLLoss()
        train_dl, valid_dl = self._split_data(X, y)
        surrogate_train_dl = SurrogateDataLoader(train_dl, self.student, self.teacher.domain_distribution, percentile=self.percentile)
        surrogate_valid_dl = SurrogateDataLoader(valid_dl, self.student, self.teacher.domain_distribution, percentile=self.percentile)
        early_stopping = EarlyStopping(patience=self.early_stopping_patience, delta=self.early_stopping_delta, epochs=self.epochs,
                                       callbacks=[(lambda i, _l, _e: self.save(model_path) if i else None)])

        for e in early_stopping:
            losses = []
            self.classifier.train()
            for x, y in surrogate_train_dl:
                optimizer.zero_grad()
                output = self.classifier(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self.classifier.eval()
            valid_losses = []
            for x, y in surrogate_valid_dl:
                loss = criterion(self.classifier(x), y)
                valid_losses.append(loss.item())
            validation_loss = sum(valid_losses)
            early_stopping.update(validation_loss)
            if verbose:
                self.logger.info(
                    f"Epoch {e}: Training Loss {sum(losses) / len(train_dl)} \t "
                    f"Validation Loss {validation_loss / len(valid_dl)}"
                )

        return self

    def _split_data(self, ts: torch.Tensor, targets: torch.Tensor) -> Tuple[DataLoader, DataLoader]:
        split_at = int(len(ts) * self.split)
        train_X, train_targets = ts[:split_at], targets[:split_at]
        valid_X, valid_targets  = ts[split_at:], targets[split_at:]
        train_ds = TensorDataset(train_X, train_targets)
        valid_ds = TensorDataset(valid_X, valid_targets)
        return DataLoader(train_ds, batch_size=self.batch_size), DataLoader(valid_ds, batch_size=self.test_batch_size)

    def save(self, path: pathlib.Path):
        with path.open("wb") as f:
            torch.save(self.classifier, f)

    @staticmethod
    def load(path: pathlib.Path) -> Classifier:
        with path.open("rb") as f:
            model = torch.load(f)
        return model
