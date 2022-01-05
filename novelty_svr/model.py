import warnings
import joblib
import tqdm
import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.base import BaseEstimator, OutlierMixin, TransformerMixin
from sklearn.metrics.pairwise import paired_distances
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    PowerTransformer,
    MinMaxScaler,
)
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import binom

from pyonlinesvr import OnlineSVR


class SlidingWindowProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size: int):
        self.window_size = window_size

    def fit(self, X: np.ndarray) -> "SlidingWindowProcessor":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        y is unused (exists for compatibility)
        """
        X = X.reshape(-1)
        # the last window would have no target to predict, e.g. for n=10: [[1, 2] -> 3, ..., [8, 9] -> 10, [9, 10] -> ?]
        new_X = sliding_window_view(X, window_shape=(self.window_size))[:-1]
        new_y = np.roll(X, -self.window_size)[: -self.window_size]
        return new_X, new_y

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        result = np.full(shape=self.window_size + len(y), fill_value=np.nan)
        result[-len(y) :] = y
        return result


class DummyScaler(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class NoveltySVR(BaseEstimator, OutlierMixin):
    def __init__(
        self,
        train_window_size: int = 16,  # D = embedding dimension
        anomaly_window_size: int = 6,  # n = event_duration (not too large)
        # confidence_level: float = 0.95,  # c \in (0, 1)
        lower_suprise_bound: Optional[int] = None,  # h = anomaly_window_size / 2
        scaling: str = "standard",  # one of "standard", "robust", "power" or empty/None
        # removes training samples from the model that are older than forgetting_time
        forgetting_time: Optional[int] = None,
        epsilon: float = 0.1,  # reused for SVR
        verbose: int = 0,  # reused for SVR
        C: float = 30.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        tol: float = 1e-3,
        stabilized: bool = True,
    ):
        self.event_duration = anomaly_window_size
        # self.confidence_level = confidence_level
        self.verbose = verbose
        self.forgetting_time = forgetting_time
        if lower_suprise_bound is None:
            self.lower_suprise_bound = anomaly_window_size // 2
        else:
            self.lower_suprise_bound = lower_suprise_bound
        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "robust":
            self.scaler = RobustScaler()
        elif scaling == "power":
            self.scaler = PowerTransformer()
        else:
            self.scaler = DummyScaler()
        self._log(f"Using {self.scaler} to scale the input data")
        self.preprocessor = SlidingWindowProcessor(window_size=train_window_size)
        self.svr = OnlineSVR(
            epsilon=epsilon,
            verbose=max(0, verbose - 2),
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            stabilized=stabilized,
            save_kernel_matrix=True,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "NoveltySVR":
        if y is not None:
            warnings.warn(
                f"y is calculated from X. Please don't pass y to NoveltySVR.fit, "
                "it will be ignored!"
            )
        if self.forgetting_time and len(X) > self.forgetting_time:
            raise ValueError(
                f"Training dataset contains too many samples ({len(X)}), "
                f"because forgetting time was set to {self.forgetting_time}."
            )

        self._log(
            f"fit(): Prepocessing data using {self.scaler} "
            f"and window size of {self.preprocessor.window_size}"
        )
        X = self.scaler.fit_transform(X)
        X, y = self.preprocessor.fit_transform(X)
        self._log(f"fit(): Input data shapes: X={X.shape}, y={y.shape}", l=2)
        self.svr.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._log(
            f"predict(): Prepocessing data using {self.scaler} and "
            f"window size of {self.preprocessor.window_size}"
        )
        X = self.scaler.transform(X)
        X, _ = self.preprocessor.transform(X)
        self._log(f"predict(): Input data shapes: X={X.shape}", l=2)
        y_hat = self.svr.predict(X)
        y_hat = self.preprocessor.inverse_transform_y(y_hat)
        self._log(f"predict(): Output data shapes: y_hat={y_hat.shape}", l=2)
        return y_hat

    def detect(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._log(
            f"detect(): Prepocessing data using {self.scaler} and "
            f"window size of {self.preprocessor.window_size}"
        )
        X = self.scaler.transform(X)
        X, y = self.preprocessor.transform(X)
        self._log(f"detect(): Input data shapes: X={X.shape}, y={y.shape}", l=2)

        self._log(f"detect(): Forecasting and online training over {len(X)} steps")
        y_hat = np.full_like(y, fill_value=np.nan)
        qs = np.zeros_like(y)
        iter = enumerate(zip(X, y))
        if self.verbose == 1 or self.verbose == 2:
            iter = tqdm.tqdm(iter, desc="Forecasting", total=len(X))

        for i, (xt, yt) in iter:
            if self._should_forget():
                self.svr.forget([0])
            y_hat[i] = self.svr.predict([xt])
            qs[i] = self._calc_current_q()
            self.svr.partial_fit([xt], [yt])

        self._log(f"detect(): Detecting novel events")
        matching_values = self._distances(y, y_hat)
        occurences = self._occurences(matching_values)
        idxs, event_scores = self._novel_events(occurences, qs)
        self._log(
            f"detect(): occurances={occurences.sum()}, novel events={len(idxs)}", l=2
        )

        self._log(f"detect(): Computing anomaly scores")
        scores = np.zeros_like(y)
        for i, es in zip(idxs, event_scores):
            scores[i : i + self.event_duration] += es
        scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).reshape(-1)
        scores = self.preprocessor.inverse_transform_y(scores)
        self._log(f"detect(): Scores shape={scores.shape}", l=2)

        if "plot" in kwargs and kwargs["plot"]:
            self._log(f"detect(): Plotting enabled - creating plots")
            self._plot(
                y,
                y_hat,
                matching_values,
                occurences,
                scores,
                skip_size="train_skip" in kwargs and kwargs["train_skip"],
            )
        return scores

    def _should_forget(self) -> bool:
        return (
            self.forgetting_time
            and self.svr._libosvr_.GetSamplesTrainedNumber() > self.forgetting_time
        )

    def _calc_current_q(self) -> float:
        if self.forgetting_time is None:
            return len(self.svr.support_) / self.svr._libosvr_.GetSamplesTrainedNumber()
        else:
            return len(self.svr.support_) / self.forgetting_time

    def _occurences(self, matching_values: np.ndarray) -> np.ndarray:
        return (matching_values > 2 * self.svr.epsilon).astype(np.int32)

    def _novel_events(self, occurences: np.ndarray, qs: np.ndarray) -> np.ndarray:
        events = sliding_window_view(occurences, window_shape=self.event_duration)
        events_norm = np.sum(events, axis=1)

        with warnings.catch_warnings():
            # use mean of empty slice returns NaN as a feature (first event won't be detected!)
            warnings.filterwarnings(
                "ignore", message="Mean of empty slice", category=RuntimeWarning
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in double_scalars",
                category=RuntimeWarning,
            )
            max_occ_bounds = np.array(
                [
                    max(self.lower_suprise_bound, events_norm[:i].mean())
                    for i in range(len(events_norm))
                ],
                dtype=np.int32,
            )
        qs = qs[: len(events_norm)]
        self._log(
            f"shapes: events={events.shape}, events_norms={events_norm.shape}, "
            f"bounds={max_occ_bounds.shape}, qs={qs.shape}",
            l=2,
        )
        densities = (
            binom(self.event_duration, events_norm)
            * np.power(qs, events_norm)
            * np.power(1 - qs, self.event_duration - events_norm)
        )
        self._log(f"densities shape={densities.shape}", l=2)
        # We need anomaly scores, therefore, we remove the confidence level threshold
        # and use the densities as event scores. The events are still filtered by
        # `max_occ_bounds`.
        # idxs = np.arange(len(events_norm))[(events_norm > max_occ_bounds) & (densities < 1 - self.confidence_level)]
        # return idxs
        event_scores = np.where(events_norm > max_occ_bounds, densities, 0.0)
        return np.arange(len(events_norm)), event_scores

    def _log(self, msg: str, l: int = 1) -> None:
        if self.verbose >= l:
            print(msg)

    def _plot(
        self, y, y_hat, matching_values, occurences, scores, skip_size: int = 0
    ) -> None:
        import matplotlib.pyplot as plt

        window_size = self.preprocessor.window_size

        def fill(a, pre=0, suf=0):
            new_a = np.full(skip_size + len(a) + pre + suf, fill_value=np.nan)
            if suf:
                new_a[skip_size + pre : -suf] = a
            else:
                new_a[skip_size + pre :] = a
            return new_a

        plt.plot(fill(y, pre=window_size), label="internal target y")
        plt.plot(fill(y_hat, pre=window_size), color="orange", label="y_hat")
        mv_plot_data = fill(matching_values, pre=window_size)
        plt.plot(mv_plot_data, color="lightgreen", label="residual")
        plt.hlines(
            y=2 * self.svr.epsilon,
            xmin=skip_size,
            xmax=len(mv_plot_data),
            color="green",
            label="epsilon",
        )
        occ_x = []
        occ_y = []
        for i, occ in enumerate(occurences):
            if occ > 0:
                occ_x.append(i + skip_size + window_size)
                occ_y.append(y[i])
        plt.scatter(occ_x, occ_y, marker="+", color="green", label="occurances")
        plt.plot(fill(scores), color="red", label="score")

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def _distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return paired_distances(x.reshape(-1, 1), y.reshape(-1, 1)).reshape(-1)

    @staticmethod
    def load(path: Path) -> "NoveltySVR":
        return joblib.load(path)
