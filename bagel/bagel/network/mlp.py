from typing import Iterable
import torch
import torch.nn as nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, net_size: Iterable[int]):
        """
        input: (batch_size, input_dims)
        output: (batch_size, output_dims), (batch_size, output_dims)
        :param input_dims:
        :param output_dims:
        :param net_size: sizes of the shared linear layers
        """
        assert input_dims > 0, "input dims must be positive: {}".format(input_dims)
        assert output_dims > 0, "output dims must be positive: {}".format(output_dims)
        super().__init__()
        last_size = input_dims
        layers = []
        for current_size in net_size:
            layers.append(nn.Linear(last_size, current_size))
            layers.append(nn.ReLU())
            last_size = current_size
        layers.append(nn.Linear(last_size, output_dims))
        self.net = nn.Sequential(*layers)

    def forward(self, _x):
        return self.net(_x)

    def penalty(self):
        """
        :return: l2 penalty on hidden layers
        """
        penalty = 0.
        for p in self.net.parameters():
            penalty += torch.sum(p ** 2)
        return penalty
