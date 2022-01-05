import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class _IndexSampler(object):
    def __init__(self, length, shuffle, drop_last, batch_size):
        self.idx = np.arange(length)
        if shuffle:
            np.random.shuffle(self.idx)
        self.pos = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.length = length

    def next(self):
        if self.pos + self.batch_size <= self.length:
            data = self.idx[self.pos: self.pos + self.batch_size]
        elif self.pos >= self.length:
            raise StopIteration()
        elif self.drop_last:
            raise StopIteration()
        else:
            data = self.idx[self.pos:]
        self.pos += self.batch_size
        return data


class KpiFrameDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size, shuffle=False, drop_last=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        #self.dataset = dataset
        self.shuffle = shuffle
        self.index_sampler = None  # type: _IndexSampler

    def __next__(self):
        return tuple(torch.from_numpy(_) for _ in self.dataset[self.index_sampler.next()])

    def __iter__(self):
        self.index_sampler = _IndexSampler(length=len(self.dataset), shuffle=self.shuffle, drop_last=self.drop_last,
                                           batch_size=self.batch_size)
        return self

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def _test_index_sampler():
    sampler = _IndexSampler(100, True, True, 11)
    try:
        while True:
            print(sampler.next())
    except StopIteration:
        pass
