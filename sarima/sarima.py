import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin

from pmdarima.arima import ARIMA, AutoARIMA
from pmdarima import model_selection

from typing import Optional


class SARIMA(BaseEstimator, OutlierMixin):
    def __init__(self,
        train_window_size: int = 500,
        prediction_window_size: int = 10,
        max_lag: Optional[int] = None,
        period: int = 1,
        max_iter: int = 20,
        exhaustive_search: bool = False,
        n_jobs: int = 1,
        fixed_orders: Optional[dict] = None
    ):
        if max_lag and train_window_size > max_lag:
            raise AttributeError("Train window size must be smaller than max lag!")

        self.train_window_size = train_window_size
        self.forecast_window_size = prediction_window_size
        self.max_lag = max_lag
        self.period = period
        self.max_iter = max_iter
        self.exhaustive_search = exhaustive_search
        self.n_jobs = n_jobs
        self.fixed_orders = fixed_orders

        # self._arima: Optional[ARIMA] = None
        # self._predictions: Optional[np.ndarray] = None
        # self._scores: Optional[np.ndarray] = None

    def _fit(self, X: np.ndarray) -> ARIMA:
        if self.fixed_orders is not None:
            # use supplied params
            seasonal = list(self.fixed_orders["seasonal_order"])
            seasonal.append(self.period)
            self.fixed_orders["seasonal_order"] = tuple(seasonal)
            print(f"Using supplied fixed orders ({self.fixed_orders}) to build SARIMA model.")
            arima = ARIMA(
                max_iter=self.max_iter,
                suppress_warnings=False,
                **self.fixed_orders
            )
        else:
            # use auto-arima to find best params
            print(f"Automatically determining SARIMA model orders using AutoARIMA (might not be optimal).")
            arima = AutoARIMA(
                start_p=1, max_p=3,
                d=None, max_d=2,
                start_q=1, max_q=3,

                seasonal=True, m=self.period,
                start_P=1, max_P=2,
                D=None, max_D=1,
                start_Q=1, max_Q=2,

                maxiter=self.max_iter,
                suppress_warnings=True, error_action="warn", trace=1,

                stepwise=not self.exhaustive_search, n_jobs=self.n_jobs,
            )

        arima.fit(X)
        print(arima.summary())
        self._arima = arima
        return arima

    def _predict(self, X: np.ndarray, with_conf_int: bool = False) -> np.ndarray:
        N = len(X)
        self._predictions = np.zeros(shape=N)
        if with_conf_int:
            self._conf_ints = np.zeros(shape=(N, 2))

        # set max_lag parameter
        self.max_lag = self.max_lag if self.max_lag else N

        # skip over training data
        i = self.train_window_size
        forecast_window_size = self.forecast_window_size
        lag_points = i
        while i < N:
            start = i
            end = i+forecast_window_size

            if lag_points >= self.max_lag:
                print(f"Recreating SARIMA model (and orders) due to max_lag (i={i} and current lag {lag_points} > {self.max_lag})")
                lag_points = 0
                self._fit(X[i - self.train_window_size:i])

            if end > N:
                end = N
                forecast_window_size = N - i

            # make forecast
            print(f"Forecasting ({start}, {end}]")
            prediction = self._arima.predict(forecast_window_size, return_conf_int=with_conf_int)
            if with_conf_int:
                y_hat, y_hat_conf = prediction
                self._predictions[start:end] = y_hat
                self._conf_ints[start:end, :] = y_hat_conf
            else:
                self._predictions[start:end] = prediction

            # update model
            self._arima.update(X[start:end])

            i += forecast_window_size
            lag_points += forecast_window_size

        return self._predictions

    def _score_points(self, X: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        self._scores = np.zeros_like(X)
        return np.abs(X - y_hat)

    def fit_predict(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        # slice training data
        train, _ = model_selection.train_test_split(X, train_size=self.train_window_size)
        self._fit(train)

        y_hat = self._predict(X)
        scores = self._score_points(X, y_hat)

        return scores


if __name__ == "__main__":
    # test parameters
    train_window_size: int = 1000
    forecast_window_size: int = 100

    period: int = 50  # must be >= 1 (if ==1: non-seasonal) (default: 1)
    max_iter: int = 10  # (default: 20)

    exhaustive_search: bool = False  # performs full grid search to find optimal SARIMA-model --> SLOW! (default: False)
    n_jobs: int = 1  # only used for grid search (default: 1)

    fixed_orders: Optional[dict] = {
        "order": (2, 0, 3),
        "seasonal_order": (0, 0, 2, period)
    }

    data = pd.read_csv("../data/dataset.csv")
    data = data.set_index("timestamp")

    model = SARIMA()
    scores = model.fit_predict(data["value"])
    predictions = model._predictions

    import matplotlib.pyplot as plt
    plt.figure()
    df = pd.DataFrame({"data": data["value"], "predictions": predictions, "scores": scores})
    df.plot()
    plt.show()
