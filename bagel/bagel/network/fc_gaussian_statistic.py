from typing import Iterable

import torch
import torch.nn as nn


class MultiLinearGaussianStatistic(nn.Module):
    """
    Generate a network with multi relu layers and a linear for mean, a linear + softplus for std
    """
    def __init__(self, input_dims: int, output_dims: int, net_size: Iterable[int], eps: float = 1e-4):
        """
        input: (batch_size, input_dims)
        output: (batch_size, output_dims), (batch_size, output_dims)
        :param input_dims:
        :param output_dims:
        :param net_size: sizes of the shared linear layers
        :param eps: std = softplus(_) + eps
        """
        assert input_dims > 0, "input dims must be positive: {}".format(input_dims)
        assert output_dims > 0, "output dims must be positive: {}".format(output_dims)
        assert eps >= 0., "epsilon must be non-negative: {}".format(eps)
        super(MultiLinearGaussianStatistic, self).__init__()
        last_size = input_dims
        shared_layers = []
        for current_size in net_size:
            shared_layers.append(nn.Linear(last_size, current_size))
            shared_layers.append(nn.ReLU())
            last_size = current_size
        self.shared_net = nn.Sequential(*shared_layers)
        self.z_mean_net = nn.Sequential(
            nn.Linear(last_size, output_dims),
        )
        self.z_std_net = nn.Sequential(
            nn.Linear(last_size, output_dims),
            nn.Softplus(),
        )
        self.eps = eps

    def forward(self, _x):
        x = self.shared_net(_x)
        z_mean = self.z_mean_net(x)
        z_std = self.z_std_net(x) + self.eps
        return z_mean, z_std

    def penalty(self):
        """
        :return: l2 penalty on hidden layers
        """
        penalty = 0.
        for p in self.shared_net.parameters():
            penalty += torch.sum(p ** 2)
        return penalty
