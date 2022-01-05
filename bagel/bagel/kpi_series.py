from functools import lru_cache
from typing import Union, Tuple
import pandas as pd
import numpy as np


class KPISeries(object):
    def __init__(self, value, timestamp, label=None, missing=None, name="", normalized: bool=False):
        self._value = np.asarray(value, np.float32)
        self._timestamp = np.asarray(timestamp, np.int64)
        self._label = np.asarray(label, np.int) if label is not None else np.zeros(np.shape(value), dtype=np.int)
        self._missing = np.asarray(missing, np.int) if missing is not None else np.zeros(np.shape(value), dtype=np.int)
        self._label[self._missing == 1] = 0

        self.normalized = normalized

        if name == "":
            import uuid
            self.name = uuid.uuid4()
        else:
            self.name = name

        self._check_shape()

        def __update_with_index(__index):
            self._timestamp = self.timestamp[__index]
            self._label = self.label[__index]
            self._missing = self.missing[__index]
            self._value = self.value[__index]

        # check interval and add missing
        __update_with_index(np.argsort(self.timestamp))
        __update_with_index(np.unique(self.timestamp, return_index=True)[1])
        intervals = np.diff(self.timestamp)
        interval = np.min(intervals)
        assert interval > 0, "interval must be positive:{}".format(interval)
        if not np.max(intervals) == interval:
            index = (self.timestamp - self.timestamp[0]) // interval
            new_timestamp = np.arange(self.timestamp[0], self.timestamp[-1] + 1, interval)
            assert new_timestamp[-1] == self.timestamp[-1] and new_timestamp[0] == self.timestamp[0]
            assert np.min(np.diff(new_timestamp)) == interval
            new_value = np.ones(new_timestamp.shape, dtype=np.float32) * self.missing_value
            new_value[index] = self.value
            new_label = np.zeros(new_timestamp.shape, dtype=np.int)
            new_label[index] = self.label
            new_missing = np.ones(new_timestamp.shape, dtype=np.int)
            new_missing[index] = self.missing
            self._timestamp, self._value, self._label, self._missing = new_timestamp, new_value, new_label, new_missing
            self._check_shape()

    def _check_shape(self):
        # check shape
        assert np.shape(self._value) == np.shape(self._timestamp) == np.shape(self._label) == np.shape(
            self._missing), "data shape mismatch, value:{}, timestamp:{}, label:{}, missing:{}".format(np.shape(self._value), np.shape(self._timestamp),
                                                                                                       np.shape(self._label), np.shape(self._missing))
        # assert self.normalized or all(self._value >= 0), "value should be non-negative"
        # if np.count_nonzero(self._missing) > 0:
        #     assert self.normalized or all(np.isclose(self._value[self._missing == 1], 0)), "Missing Value should be zero:{}".format(np.unique(self.value[self.missing == 1]))

    @property
    def value(self):
        return self._value

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def label(self):
        return self._label

    @property
    def missing(self):
        return self._missing

    @property
    def time_range(self):
        from datetime import datetime
        return datetime.fromtimestamp(np.min(self.timestamp)), datetime.fromtimestamp(np.max(self.timestamp))

    @property
    def length(self):
        return np.size(self.value, 0)

    @property
    def abnormal(self):
        return np.logical_or(self.missing, self.label).astype(np.int)

    @property
    def missing_rate(self):
        return float(np.count_nonzero(self.missing)) / float(self.length)

    @property
    def anormaly_rate(self):
        return float(np.count_nonzero(self.label)) / float(self.length)

    @lru_cache()
    def normalize(self, mean=None, std=None, return_statistic=False):
        """
        :param return_statistic: return mean and std or not
        :param std: optional, normalize by given std
        :param mean: optional, normalize by given mean
        :param inplace: inplace normalize
        :return: data_set, mean, std
        """
        mean = np.mean(self.value) if mean is None else mean
        std = np.std(self.value) if std is None else std
        normalized_value = (self.value - mean) / np.clip(std, 1e-4, None)
        target = KPISeries(normalized_value, self.timestamp, self.label, self.missing, normalized=True, name=self.name)
        if return_statistic:
            return target, mean, std
        else:
            return target

    def split(self, radios: Union[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], Tuple[float, float, float]]) -> \
    Tuple['KPISeries', 'KPISeries', 'KPISeries']:
        """
        :param radios: radios of each part, eg. [0.2, 0.3, 0.5] or [(0, 0.1), (0, 0.2), (0.5, 1.0)]
        :return: tuple of DataSets
        """
        if np.asarray(radios).ndim == 1:
            radios = radios  # type: Tuple[float, float, float]
            assert abs(1.0 - sum(radios)) < 1e-4
            split = np.asarray(np.cumsum(np.asarray(radios, np.float64)) * self.length, np.int)
            split[-1] = self.length
            split = np.concatenate([[0], split])
            result = []
            for l, r in zip(split[:-1], split[1:]):
                result.append(KPISeries(self.value[l:r], self.timestamp[l:r], self.label[l:r], self.missing[l:r]))
        elif np.asarray(radios).ndim == 2:
            radios = radios  # type: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
            result = []
            for start, end in radios:
                si = int(self.length * start)
                ei = int(self.length * end)
                result.append(KPISeries(self.value[si:ei], self.timestamp[si:ei], self.label[si:ei], self.missing[si:ei]))
        else:
            raise ValueError("split radios in wrong format: {}".format(radios))
        ret = tuple(result)  # type: Tuple
        return ret

    @lru_cache()
    def label_sampling(self, sampling_rate: float = 1.):
        """
        sampling label by segments
        :param sampling_rate: keep at most sampling_rate labels
        :return:
        """
        sampling_rate = float(sampling_rate)
        assert 0. <= sampling_rate <= 1., "sample rate must be in [0, 1]: {}".format(sampling_rate)
        if sampling_rate == 1.:
            return self
        elif sampling_rate == 0.:
            return KPISeries(value=self.value, timestamp=self.timestamp, label=None, missing=self.missing, name=self.name, normalized=self.normalized)
        else:
            target = np.count_nonzero(self.label) * sampling_rate
            label = np.copy(self.label).astype(np.int8)
            anormaly_start = np.where(np.diff(label) == 1)[0] + 1
            if label[0] == 1:
                anormaly_start = np.concatenate([[0], anormaly_start])
            anormaly_end = np.where(np.diff(label) == -1)[0] + 1
            if label[-1] == 1:
                anormaly_end = np.concatenate([anormaly_end, [len(label)]])

            x = np.arange(len(anormaly_start))
            np.random.shuffle(x)

            for i in range(len(anormaly_start)):
                idx = np.asscalar(np.where(x == i)[0])
                label[anormaly_start[idx]:anormaly_end[idx]] = 0
                if np.count_nonzero(label) <= target:
                    break
            return KPISeries(value=self.value, timestamp=self.timestamp, label=label, missing=self.missing, name=self.name, normalized=self.normalized)

    @property
    def missing_value(self):
        return self.value[self.missing == 1][0] if np.count_nonzero(self.missing) > 0 else 2 * np.min(self.value) - np.max(self.value)

    def __add__(self, other):
        """
        :type other KPISeries
        :param other:
        :return:
        """
        if not isinstance(other, type(self)):
            raise ValueError("Only KpiSeries can be added together")
        value = np.concatenate([self.value, other.value])
        missing = np.concatenate([self.missing, other.missing])
        if len(np.unique(value[missing == 1])) != 1:
            value[missing == 1] = 2 * np.min(value) - np.max(value)
        timestamp = np.concatenate([self.timestamp, other.timestamp])
        label = np.concatenate([self.label, other.label])
        return KPISeries(timestamp=timestamp, value=value, label=label, missing=missing, name=self.name, normalized=self.normalized)

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            raise ValueError("Only KPISeries can be added together")
        value = np.concatenate([self.value, other.value])
        missing = np.concatenate([self.missing, other.missing])
        if len(np.unique(value[missing == 1])) != 1:
            value[missing == 1] = 2 * np.min(value) - np.max(value)
        timestamp = np.concatenate([self.timestamp, other.timestamp])
        label = np.concatenate([self.label, other.label])
        self._timestamp, self._value, self._label, self._missing = timestamp, value, label, missing
        self._check_shape()
        return self

    def __len__(self):
        return len(self.timestamp)

    @staticmethod
    def dump(kpi, path: str, **kwargs):
        import pandas as pd
        df = pd.DataFrame()
        df["timestamp"] = kpi.timestamp
        df["value"] = kpi.value
        df["label"] = kpi.label
        df["missing"] = kpi.missing
        if path.endswith(".csv"):
            df.to_csv(path, **kwargs)
        elif path.endswith(".hdf"):
            df.to_csv(path, **kwargs)
        else:
            raise ValueError(f"Unknown format. Csv and hdf are supported, but given {path}")

    @staticmethod
    def load(path: str, **kwargs):
        import pandas as pd
        import os
        if path.endswith(".csv"):
            df = pd.read_csv(path, **kwargs)
        elif path.endswith(".hdf"):
            df = pd.read_hdf(path, **kwargs)
        else:
            raise ValueError(f"Unknown format. Csv and hdf are supported, but given {path}")
        return KPISeries(timestamp=df["timestamp"],
                         value=df["value"],
                         missing=df["missing"] if "missing" in df else None,
                         label=df["label"] if "label" in df else None,
                         name=os.path.basename(path)[:-4])

    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame, name):
        df = dataframe
        return KPISeries(timestamp=df["timestamp"],
                         value=df["value"],
                         missing=df["missing"] if "missing" in df.columns else None,
                         label=df["label"] if "label" in df.columns else None, name=name)
