import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TAnoGANDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_length, stride: int):
        super(Dataset).__init__()

        x = X.astype("float")
        y = y.astype("int")

        self.window_length = window_length

        self.stride = stride
        self.n_feature = x.shape[1]

        # standardize
        x = StandardScaler().fit_transform(x.reshape(-1, self.n_feature))
        x, y = self.unroll(x, y)
        self._length = len(x)

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        return self.x[key], self.y[key]

    def unroll(self, data, labels):
        un_data = []
        un_labels = []
        seq_len = self.window_length
        stride = self.stride

        idx = 0
        while(idx < len(data) - seq_len):
            un_data.append(data[idx:idx+seq_len])
            un_labels.append(labels[idx:idx+seq_len])
            idx += stride
        return np.array(un_data), np.array(un_labels)
