import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Detector():
    def __init__(self):
        pass

    def detect(self, predictedY: torch.Tensor, test_dataset: Dataset) -> np.ndarray:
        _, test_y = next(iter(DataLoader(test_dataset, batch_size=predictedY.shape[0])))

        # calculate euclidian distance
        anomaly_score = torch.sqrt(F.mse_loss(predictedY.detach(), test_y.detach(), reduction="none").sum(dim=[1, 2]))
        # standardize error
        anomaly_score = (anomaly_score - anomaly_score.mean()).abs() / anomaly_score.std()
        return anomaly_score.numpy()
