from typing import Sequence
import numpy as np
from torch.utils.data import Dataset
from .kpi_series import KPISeries


class KpiFrameDataset(Dataset):
    def __init__(self, kpi: KPISeries, frame_size: int, missing_injection_rate: float = 0.0):
        self._kpi = kpi
        self._frame_size = frame_size

        # self._strided_value = self.to_frames(self.normalized(kpi.value), frame_size)
        self._strided_value = self.to_frames(kpi.value, frame_size)
        self._strided_abnormal = self.to_frames(kpi.abnormal, frame_size)
        self._strided_missing = self.to_frames(kpi.missing, frame_size)
        self._strided_label = self.to_frames(kpi.label, frame_size)
        self._missing_injection_rate = missing_injection_rate
        self._missing_value = kpi.missing_value

    def __len__(self):
        return np.size(self._strided_value, 0)

    def __getitem__(self, item):
        value = np.copy(self._strided_value[item]).astype(np.float32)
        normal = 1 - np.copy(self._strided_abnormal[item]).astype(np.int)
        label = np.copy(self._strided_label[item]).astype(np.int)

        _missing_injection(value, normal=normal, label=label, missing_value=self._missing_value,
                           missing_injection_rate=self._missing_injection_rate)
        return value.astype(np.float32), normal.astype(np.float32)

    @staticmethod
    def to_frames(array, frame_size: int = 120):
        # noinspection PyProtectedMember
        from numpy.lib.stride_tricks import as_strided
        array = as_strided(array, shape=(np.size(array, 0) - frame_size + 1, frame_size),
                           strides=(array.strides[-1], array.strides[-1]))
        return array


def _missing_injection(value, normal, label, missing_value, missing_injection_rate):
    injected_missing = np.random.binomial(1, missing_injection_rate, np.shape(value[normal == 1]))
    normal[normal == 1] = 1 - injected_missing
    value[np.logical_and(normal == 0, label == 0)] = missing_value
    return value, normal


class TimestampDataset(KpiFrameDataset):
    TS_OFFSET = 8 * 3600

    def __init__(self, kpi: KPISeries, frame_size: int):
        super().__init__(kpi, frame_size, missing_injection_rate=0.0)
        self._timestamp_feature = self.normalize(self._kpi.timestamp)
        self._timestamp_digits = self.digits(self._kpi.timestamp)
        self._minute_feature = self.normalize(self.ts2minute(self._kpi.timestamp))
        self._hour_feature = self.normalize(self.ts2hour(self._kpi.timestamp))
        self._day_of_week_feature = self.normalize(self.ts2day_of_week(self._kpi.timestamp))
        self._day_in_year_feature = self.normalize(self.ts2day_in_year(self._kpi.timestamp))

        self._one_hot_minute = self.one_hot(self.ts2minute(self._kpi.timestamp), width=60, loc=0)
        self._one_hot_hour = self.one_hot(self.ts2hour(self._kpi.timestamp), width=24, loc=0)
        self._one_hot_day_of_week = self.one_hot(self.ts2day_of_week(self._kpi.timestamp), width=7, loc=0)
        self._one_hot_month = self.one_hot(self.ts2month(self._kpi.timestamp), width=12, loc=0)
        self._one_hot_year = self.one_hot(self.ts2year(self._kpi.timestamp))

    def __getitem__(self, item):
        index = np.asarray(item) + self._frame_size - 1
        timestamp = self._kpi.timestamp[index]
        # ret = np.concatenate([self.one_hot(self.ts2hour(timestamp), width=24), ], axis=-1).astype(np.float32)
        hourly_feature = self._one_hot_hour[index]
        day_of_week_feature = self._one_hot_day_of_week[index]
        month_feature = self._one_hot_month[index]
        year_feature = self._one_hot_year[index]
        timestamp_feature = np.expand_dims(
            (timestamp - np.min(self._kpi.timestamp)) / (np.max(self._kpi.timestamp) - np.min(self._kpi.timestamp)), -1)
        # ret = np.concatenate([, hourly_feature, day_of_week_feature], axis=-1).astype(np.float32)
        ret = np.concatenate([
            self._one_hot_minute[index],
            self._one_hot_hour[index],
            self._one_hot_day_of_week[index],
            # self._timestamp_digits[index],
            # np.expand_dims(self._day_in_year_feature[index], -1),
            # self._one_hot_year[index],
        ],
            axis=-1).astype(np.float32)
        # ret = np.concatenate([timestamp_feature, ], axis=-1).astype(np.float32)
        return ret

    @staticmethod
    def ts2day_of_week(_ts):
        ts = np.asarray(_ts) + TimestampDataset.TS_OFFSET
        return ((ts // 86400) + 4) % 7

    @staticmethod
    def ts2day_in_year(_ts):
        ts = np.asarray(_ts) + TimestampDataset.TS_OFFSET
        return (ts // 86400) % 365

    @staticmethod
    def ts2hour(_ts):
        ts = np.asarray(_ts) + TimestampDataset.TS_OFFSET
        return (ts % 86400) // 3600

    @staticmethod
    def ts2minute(_ts):
        ts = np.asarray(_ts) + TimestampDataset.TS_OFFSET
        return ((ts % 86400) % 3600) // 60

    @staticmethod
    def ts2month(_ts):
        ts = np.asarray(_ts) + TimestampDataset.TS_OFFSET
        return np.minimum(11, ((ts // 86400) % 365) // 30)

    @staticmethod
    def ts2year(_ts):
        ts = np.asarray(_ts) + TimestampDataset.TS_OFFSET
        return (ts // 86400) // 365 + 1970

    @staticmethod
    def normalize(arr):
        scale = np.max(arr) - np.min(arr)
        if np.abs(scale - 0) < 1e-3:
            return np.ones_like(arr)
        else:
            return (arr - np.min(arr)) / scale

    @staticmethod
    def one_hot(_array, width=None, loc=0):
        array = np.copy(np.asarray(_array)).astype(np.int)  # type: np.ndarray
        assert np.ndim(array) == 1 or np.ndim(array) == 0, "only 1d array or scalar is supported, shape is {}".format(
            np.shape(array))
        if width is None:
            width = np.max(array) - np.min(array) + 1
            loc = np.min(array)
        output = np.zeros(np.shape(array) + (width,), dtype=np.int)
        if np.ndim(array) == 1:
            output[np.arange(0, np.size(array)), array - loc] = 1
        else:
            output[array - loc] = 1
        return output

    @staticmethod
    def digits(_array, width=None):
        if width is None:
            width = int(np.max(np.floor(np.log(_array) / np.log(10)) + 1))
        _ = np.copy(_array).astype(np.int)
        arrays = []
        for i in range(width):
            arrays.append(np.expand_dims(_ % 10, -1))
            _ //= 10
        return np.concatenate(arrays, -1)
