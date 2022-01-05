import numpy.typing as npt
from typing import Optional, Union

from .fuzzy import Sugeno, Choquet, WrappedFCM

import pomegranate as pg

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


class MultiHMM(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.hmm: Optional[Union[pg.HiddenMarkovModel, str]] = None

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> 'MultiHMM':
        y = ['None-start'] + y.tolist() + ['None-end']
        self.hmm = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution, n_components=2, X=X)
        self.hmm.bake()

        self.hmm.fit(X, labels=y, algorithm="viterbi")
        return self

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        log_proba = self.hmm.predict_log_proba(X)
        return log_proba[:, 1]

    def __getstate__(self):
        self.hmm = self.hmm.to_json()
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.hmm = pg.HiddenMarkovModel.from_json(self.hmm)


class MultiHMMADBuilder:
    def __init__(self, n_bins: int, discretizer: str, n_features: int):
        self.n_bins = n_bins
        self.discretizer = discretizer
        self.n_features = n_features

    def _build_discretizer(self):
        if self.n_features == 1:
            return KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')

        if self.discretizer == "sugeno":
            return Pipeline([
                ("Sugeno", Sugeno(axis=1)),
                ("Discretization", KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform'))
            ])
        elif self.discretizer == "choquet":
            return Pipeline([
                ("Choquet", Choquet(axis=1)),
                ("Discretization", KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform'))
            ])
        else:  # if self.discretizer == "fcm"
            return WrappedFCM(n_bins=self.n_bins)

    def build(self) -> Pipeline:
        algorithm = Pipeline([
            ("StandardScaler", StandardScaler()),
            ("Discretizer", self._build_discretizer()),
            ("MultiHMM", MultiHMM())
        ])

        return algorithm


def train():
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("../../data/dataset.csv")
    y = df.values[:, -1].astype(int).astype(str)
    X = df.values[:, 1:-1]

    alphabet_size = 20

    single = KBinsDiscretizer(n_bins=alphabet_size, encode='ordinal', strategy='uniform')

    sugeno = Pipeline([
        ("Sugeno", Sugeno(axis=1)),
        ("Discretization", KBinsDiscretizer(n_bins=alphabet_size, encode='ordinal', strategy='uniform'))
    ])

    choquet = Pipeline([
        ("Choquet", Choquet(axis=1)),
        ("Discretization", KBinsDiscretizer(n_bins=alphabet_size, encode='ordinal', strategy='uniform'))
    ])

    fcm = WrappedFCM(n_bins=alphabet_size)

    algorithm = Pipeline([
        ("StandardScaler", StandardScaler()),
        ("Reducer", single),
        ("MultiHMM", MultiHMM())
    ])

    algorithm.fit(X[:1000], y[:1000])
    scores = algorithm.predict(X[1000:])

    plt.plot(scores, label="scores")
    plt.plot(X[1000:] / X[1000:].max(), label="anomalies")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()
