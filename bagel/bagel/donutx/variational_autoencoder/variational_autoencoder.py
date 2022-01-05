import torch.nn as nn
from torch.autograd.variable import Variable


class VariationalAutoencoder(nn.Module):
    def forward(self, *, n_sample=1, **kwargs):
        raise NotImplementedError("Don't call base class")

    def __init__(self, variational: nn.Module, generative: nn.Module, n_sample=1):
        super().__init__()
        self.variational = variational
        self.generative = generative
        self.n_sample = n_sample

    def penalty(self) -> Variable:
        return self.variational.penalty() + self.generative.penalty()


VAE = VariationalAutoencoder
