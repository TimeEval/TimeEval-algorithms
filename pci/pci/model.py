import numpy as np
from scipy import stats
from typing import Tuple


class PCIAnomalyDetector:
    def __init__(self, k: int, p: float, calculate_labels=True):
        assert 0 < p < 1, "p must be between 0 and 1"

        self.k = k
        self.p = p
        self.w = np.concatenate((np.arange(1, k+1), np.arange(1, k+1)[::-1]))
        self.calculate_labels = calculate_labels

    def _pci(self, v: float, window_predictions: np.ndarray, eta: np.ndarray) -> Tuple[float, float]:
        t = stats.t.ppf(self.p, df=2 * self.k - 1)
        s = (eta - window_predictions).std()
        lower_bound = v - t * s * np.sqrt(1 + (1 / (2 * self.k)))
        upper_bound = v + t * s * np.sqrt(1 + (1 / (2 * self.k)))
        return lower_bound, upper_bound

    def _predict(self, eta: np.ndarray) -> float:
        eta_no_nan = eta[~np.isnan(eta)]
        w_no_nan = self.w[~np.isnan(eta)]
        v_hat = np.matmul(eta_no_nan, w_no_nan) / w_no_nan.sum()
        return v_hat

    def _generate_window(self, ts: np.ndarray, i: int) -> np.ndarray:
        result = np.zeros(2 * self.k)
        m = len(ts)
        left_start = max(i - self.k, 0)
        left_end = max(i - 1 + 1, 0)
        left_length = left_end - left_start
        right_start = min(i + 1, m)
        right_end = min(i + self.k + 1, m)
        right_length = right_end - right_start

        result[self.k - left_length:self.k] = ts[left_start:left_end]
        result[self.k:self.k + right_length] = ts[right_start:right_end]
        return result

    def _combine_left_right(self, left: list, right: list) -> np.ndarray:
        prediction_window = np.zeros(2 * self.k)
        if len(left) > 0:
            prediction_window[self.k - len(left):self.k] = left
        prediction_window[self.k:self.k + len(right)] = right
        return prediction_window

    def detect(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        anomaly_scores = np.zeros(len(ts))
        anomaly_labels = np.zeros(len(ts), dtype=np.int)
        m = len(ts)
        left_predictions = []
        right_predictions = []
        for i in range(m):
            v = ts[i]

            start = i + len(right_predictions)
            for j in range(start, min(start+self.k, m)+1):
                eta_ = self._generate_window(ts, j)
                right_v = self._predict(eta_)
                right_predictions.append(right_v)

            v_hat = right_predictions.pop(0)
            predictions.append(v_hat)

            anomaly_scores[i] = abs(v_hat - v)

            if self.calculate_labels:
                prediction_window = self._combine_left_right(left_predictions, right_predictions)
                eta = self._generate_window(ts, i)
                lower_bound, upper_bound = self._pci(v_hat, prediction_window, eta)
                anomaly_labels[i] = int(not lower_bound < v_hat < upper_bound)

            left_predictions.append(v_hat)
            if len(left_predictions) > self.k:
                del left_predictions[0]

        return anomaly_scores, anomaly_labels
