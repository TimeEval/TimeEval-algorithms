import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Tuple


class Reconstruction(nn.Module):
    def __init__(self, channel_sizes: List[int], latent_size: int, num_features: int):
        super().__init__()

        self.encoder = Encoder(channel_sizes, latent_size)
        self.decoder = Decoder(channel_sizes, latent_size, num_features)

        self.loss = CustomLoss()

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, std = self.encoder(X)
        y = self.decoder(mean, std)
        return y, mean, std


class Encoder(nn.Module):
    def __init__(self, channel_sizes: List[int], latent_size: int):
        super().__init__()

        self.fcs = nn.ModuleList(
            nn.Linear(in_features=channel_sizes[c], out_features=channel_sizes[c+1]) for c in range(len(channel_sizes)-1)
        )
        self.mean = nn.Linear(channel_sizes[-1], latent_size)
        self.std = nn.Linear(channel_sizes[-1], latent_size)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        for fc in self.fcs:
            X = torch.relu(fc(X))
        mean = self.mean(X)
        std = torch.relu(self.std(X))
        return mean, std


class Decoder(nn.Module):
    def __init__(self, channel_sizes: List[int], latent_size: int, num_features: int):
        super().__init__()

        channel_sizes = [latent_size]+ list(reversed(channel_sizes)) + [num_features]

        self.fcs = nn.ModuleList(
            nn.Linear(in_features=channel_sizes[c], out_features=channel_sizes[c + 1]) for c in
            range(len(channel_sizes) - 1)
        )

    def forward(self, mean: Tensor, std: Tensor) -> Tensor:
        X = torch.distributions.Normal(mean, std).sample()
        for fc in self.fcs:
            X = torch.relu(fc(X))
        return X


class CustomLoss(nn.MSELoss):
    def _kl_div(self, mu: Tensor, std: Tensor) -> Tensor:
        loss = -0.5 * torch.sum(1 + torch.log(std+1e-6) - torch.pow(mu, 2) - std)
        return loss

    def forward(self, input: Tensor, target: Tensor, mu: Tensor, std: Tensor) -> Tensor:
        loss = super().forward(input, target) + self._kl_div(mu, std) / input.shape[0]
        return loss
