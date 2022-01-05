import numpy as np
from scipy import signal
import torch

class ScalogramEncoding:
    @staticmethod
    def encode(window: np.ndarray, img_size: int) -> torch.tensor: 
        scales = 2**(np.arange(1, img_size + 1) / 4)
        cwtmatr = signal.cwt(window, signal.ricker, scales)
        img = torch.nn.AvgPool1d(len(window) // img_size)(torch.unsqueeze(torch.tensor(cwtmatr), 0))
        return img.float()
