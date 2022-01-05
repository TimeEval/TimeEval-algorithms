import numpy as np
from statsmodels.tsa.holtwinters.model import ExponentialSmoothing, HoltWintersResults
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)


class TripleES:
    def __init__(self,
                 ts: np.ndarray,
                 train_window_size: int,
                 period: int,
                 trend: str,
                 seasonality: str):
        self.ts = ts
        assert train_window_size >= 10 + period, \
           """Cannot use heuristic method to compute initial seasonal and levels with less than periods + 10 datapoints."""
        self.window_size = train_window_size
        self.seasonal_periods = period
        self.trend = trend
        self.seasonality = seasonality

    def fit_window(self, X) -> HoltWintersResults:
        model = ExponentialSmoothing(X,
                                     seasonal_periods=self.seasonal_periods,
                                     trend=self.trend,
                                     seasonal=self.seasonality,
                                     use_boxcox=True,
                                     initialization_method="estimated")
        return model.fit()

    def predict(self, model: HoltWintersResults) -> np.float64:
        return model.forecast(1)[0]

    def fit_predict(self, plot: bool = False) -> np.ndarray:
        if plot:
            pred = np.full_like(self.ts, fill_value=np.nan)
            stds = np.full_like(self.ts, fill_value=np.nan)

        scores = np.zeros_like(self.ts)
        for i in range(len(self.ts) - self.window_size):
            # select window of data and y to predict
            X = self.ts[i:i+self.window_size]
            y = self.ts[i+self.window_size]

            # fit model to window
            model_window: HoltWintersResults = self.fit_window(X)
            X_pred = model_window.fittedvalues

            # predict next value
            y_hat = self.predict(model_window)
            err = np.abs(y - y_hat)

            # calculate residuals
            residuals = (X - X_pred)
            stdev = np.nanstd(residuals)

            # calculate anomaly score for y
            score_y = err / stdev
            scores[i+self.window_size] = score_y

            if plot:
                pred[i+self.window_size] = y_hat
                stds[i+self.window_size] = 3*stdev

        if plot:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, sharex="col", figsize=(20, 8))
            index = np.arange(len(self.ts))
            axs[0].plot(index, self.ts, label="TS")
            axs[0].plot(index, pred, label="Pred", color="orange")
            axs[0].fill_between(index, pred-stds, pred+stds, label="Pred (confidence: 3sigma)", color="orange", alpha=0.2)
            axs[0].legend()
            axs[1].plot(index, scores, label="Scores")
            axs[1].legend()
            fig.savefig("/results/triple_es-plot.pdf")

        return scores
