import numpy as np
import bottleneck as bn


class MedianMethod:
    """
    :param np.ndarray timeseries: univariate timeseries
    :param int neighbourhood_size: number of time steps to include in the window from past and future

    example: [1, 2, 6, 4, 5] with neighbourhood_size of 1 
    move_median creates windows like this: [nan, nan, 2, 4, 4] 
    We want the indexes of the timeseries to align with the window median
    So we shift backwards like this using np.roll: [nan, 2, 4, 4, nan]
    This way we can calculate accurate differences between the timeseries data points
    and the median of their neighbourhood:
      [  1, 2, 6, 4, 5  ]
    - [nan, 2, 4, 4, nan]
    = [nan, 0, 2, 0, nan]
    """

    def __init__(self, timeseries, neighbourhood_size):
        self._timeseries = timeseries
        self._neighbourhood_size = neighbourhood_size

    def compute_windows(self, type):
        if type == "std":
            windows = bn.move_std(self._timeseries, window=self._neighbourhood_size*2 + 1)
        else:
            windows = bn.move_median(self._timeseries, window=self._neighbourhood_size*2 + 1)
        return np.roll(windows, -self._neighbourhood_size)

    def fit_predict(self):
        median_windows = self.compute_windows("median")
        std_windows = self.compute_windows("std")
        dist_windows = np.absolute(median_windows - self._timeseries)
        scores = dist_windows / std_windows
        return np.nan_to_num(scores)
