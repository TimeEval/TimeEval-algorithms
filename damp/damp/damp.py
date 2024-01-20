from __future__ import annotations

from typing import Optional, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import stumpy as st
import pandas as pd
import logging

from .preprocessor import DAMPPreprocessor


def nextpow2(x: int) -> int:
    return int(np.ceil(np.log2(x)))


class DAMP(BaseEstimator, TransformerMixin):
    """
    Implementation of the DAMP algorithm that is described in [this paper](https://www.cs.ucr.edu/~eamonn/DAMP_long_version.pdf).
    """

    def __init__(self,
                 m: int = 50,
                 sp_index: int = 200,
                 x_lag: Optional[int] = None,
                 golden_batch: Optional[np.ndarray] = None,
                 preprocessing: bool = True,
                 lookahead: Optional[int] = None,
                 with_prefix_handling: bool = True):
        super().__init__()
        self.m = m
        self.sp_index = sp_index
        self.x_lag = x_lag or 2**nextpow2(8*self.m)
        self.golden_batch = golden_batch
        self.preprocessing = preprocessing
        self.lookahead = int(2**nextpow2(lookahead) if lookahead is not None else 2 ** nextpow2(16 * self.m))
        self.with_prefix_handling = with_prefix_handling

        self._pv: Optional[np.ndarray] = None
        self._amp: Optional[np.ndarray] = None
        self._bsf = 0

    def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> np.ndarray:
        self._validate_sp_index(X)
        if self.preprocessing:
            preprocessor = DAMPPreprocessor(m=self.m, sp_index=self.sp_index)
            X = preprocessor.fit_transform(X)
        self._pv = np.ones(len(X) - self.m + 1, dtype=int)
        self._amp = np.zeros_like(self._pv, dtype=float)

        if self.with_prefix_handling:
            self._handle_prefix(X)

        for i in range(self.sp_index, len(X) - self.m + 1):
            if not self._pv[i]:
                self._amp[i] = self._amp[i-1]-0.00001
                continue

            self._amp[i] = self._backward_processing(X, i)
            self._forward_processing(X, i)

        return self._amp

    def partial_fit(self, X, y=None, **fit_params) -> DAMP:
        raise NotImplementedError("This is not yet supported")

    def _backward_processing(self, X: np.ndarray, i) -> float:
        amp_i = np.inf
        prefix = 2**nextpow2(self.m)
        max_lag = min(self.x_lag or i, i)
        reference_ts = self.golden_batch or X[i-max_lag:i]
        first = True
        expansion_num = 0

        while amp_i >= self._bsf:
            if prefix >= max_lag:  # search reaches the beginning of the time series
                amp_i = min(self._distance(X[i:i+self.m], reference_ts))
                if amp_i > self._bsf:
                    self._bsf = amp_i
                break
            else:
                if first:
                    first = False
                    amp_i = min(self._distance(X[i:i+self.m], reference_ts[-prefix:]))
                else:
                    start = i-max_lag+(expansion_num * self.m)
                    end = int(i-(max_lag/2)+(expansion_num * self.m))
                    amp_i = min(self._distance(X[i:i+self.m], X[start:end]))

                if amp_i < self._bsf:
                    break
                else:
                    prefix = 2*prefix
                    expansion_num += 1

        return amp_i

    def _forward_processing(self, X: np.ndarray, i):
        start = i + self.m
        end = start + self.lookahead
        indices: List[int] = []

        if end < len(X):
            d = self._distance(X[i:i+self.m], X[start:end])
            indices = np.argwhere(d < self._bsf)
            indices += start

        self._pv[indices] = 0

    def _distance(self, Q: np.ndarray, T: np.ndarray) -> np.ndarray:
        n_variates = Q.shape[1]
        return np.sum([st.core.mass(Q[:, d], T[:, d]) for d in range(n_variates)], axis=0)

    def _validate_sp_index(self, X: np.ndarray):
        if self.sp_index / self.m < 4:
            logging.warning(f"The training sequence is recommended to be at least 4 times of the window size 'm'. "
                            f"m ({self.m}) x 4 = sp_index ({self.m*4})")
            if self.sp_index < self.m:
                logging.warning(f"The training sequence must be at least the window size 'm'. `sp_index` has been set to {self.m}")
                self.sp_index = self.m
        elif self.sp_index > (X.shape[0] - self.m + 1):
            logging.warning(f"The training sequence cannot be greater than `length(X) - sp_index + 1`."
                            f"`sp_index` has been set to {X.shape[0] - self.m + 1}")

    def _handle_prefix(self, X: np.ndarray):
        """
        From authors' implementation. Not described in paper.
        """
        for i in range(self.sp_index, min(self.sp_index + (16 * self.m), self._pv.shape[0])):
            if self._pv[i] != 0:
                self._amp[i] = self._amp[i-1]-0.00001
                continue

            if i + self.m > X.shape[0]:
                break

            query = X[i:i+self.m]
            self._amp[i] = min(self._distance(query, X[:i]))
            self._bsf = max(self._amp)

            if self.lookahead > 0:
                start_of_mass = min(i+self.m, X.shape[0])
                end_of_mass = min(start_of_mass+self.lookahead, X.shape[0])

                if (end_of_mass - start_of_mass + 1) > self.m:
                    distance_profile = self._distance(query, X[start_of_mass:end_of_mass])
                    dp_index_less_than_BSF = np.argwhere(distance_profile < self._bsf)
                    ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass - 1
                    self._pv[ts_index_less_than_BSF] = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T = pd.read_csv("../../data/dataset.csv").iloc[:, 1:-1].values
    print(T)
    print("Fitting DAMP")
    damp = DAMP(m=250, sp_index=1000)
    mp = damp.fit_transform(T)

    plt.plot(T/T.max(), label="TS")
    plt.plot(mp/mp.max(), label="MP")
    plt.legend()
    plt.show()
