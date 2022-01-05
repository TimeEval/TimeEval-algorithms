from typing import List, Sequence

import numpy as np
from torch.utils.data import Dataset


class DstackDataset(Dataset):
    """
    Return data in several datasets in depth stack.
    eg.
        A return data in sample shape (10, ), B return data in sample shape (2, ),
        then ParallelDataset return data in sample shape (12, )
        All the sub-datasets are supposed to have the same length

    At most ONE dataset can return more than one items
    eg.
        A return (10, ), (10, ), B return (4, )
        then this will return (14, ), (10, )
    """
    def __init__(self, datasets: List[Dataset], complex_dataset=None):
        dataset_lengths = list(len(_) for _ in datasets)
        assert len(set(dataset_lengths)) == 1, "length of these datasets must be all the same: {}".format(dataset_lengths)
        self._length = dataset_lengths[0]
        self._datasets = datasets

        self._complex_dataset = complex_dataset

    def __getitem__(self, index):
        a =  np.concatenate([dataset[index] if dataset is not self._complex_dataset else dataset[index][0] for dataset in self._datasets], axis=-1)
        if self._complex_dataset is None:
            return a
        else:
            return (a, ) + self._complex_dataset[index][1:]

    def __len__(self):
        return self._length


class VstackDataset(Dataset):
    """
    Return data in several datasets in vertical stack.
    eg.
        A return data in sample shape (10, ), B return data in sample shape (2, ),
        then ParallelDataset return data in sample shape (10, ), (2, )
        All the sub-datasets are supposed to have the same length

    Dataset __getitem__ return type can't be tuple unless returning multiple array
    """
    def __init__(self, datasets: List[Dataset]):
        dataset_lengths = list(len(_) for _ in datasets)
        assert len(set(dataset_lengths)) == 1, "length of these datasets must be all the same: {}".format(dataset_lengths)
        self._length = dataset_lengths[0]
        self._datasets = datasets

    def __getitem__(self, index):
        x =  [dataset[index] if isinstance(dataset[index], tuple) else (dataset[index],) for dataset in self._datasets]
        return sum(x, tuple())

    def __len__(self):
        return self._length
