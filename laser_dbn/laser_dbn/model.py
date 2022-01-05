import pomegranate as pg
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.preprocessing import KBinsDiscretizer
from typing import List, Optional, Tuple, Union
from collections import Counter
import networkx as nx
import joblib
from os import PathLike


class DynamicBayesianNetworkAD(BaseEstimator, OutlierMixin):
    """
    # Dynamic-Bayesian-Network Anomaly-Detector

    - Discretization
    - (default) Two-Timeslice BN architecture
    """

    def __init__(self, timesteps: int, discretizer_n_bins: int):
        self.features: List[Tuple[int, int]] = []
        self.timesteps = timesteps
        self.discretizer = KBinsDiscretizer(n_bins=discretizer_n_bins, encode='ordinal', strategy='uniform')
        self.bayesian_network: Optional[Union[pg.BayesianNetwork, str]] = None

    def _get_distribution(self, X: np.ndarray) -> pg.DiscreteDistribution:
        distributed = {str(k): float(v) / len(X) for k, v in Counter(X).items()}
        return pg.DiscreteDistribution(distributed)

    def _preprocess_data(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = self.discretizer.fit_transform(X).astype(int) if fit else self.discretizer.transform(X).astype(int)
        rolled_Xs = [X]
        for t in range(1, self.timesteps):
            rolled_Xs.append(np.roll(X, -t, axis=0))
        X = np.concatenate(rolled_Xs, axis=1)[:-(self.timesteps-1)]
        return X

    def _build_contraint_graph(self, n_nodes: int) -> nx.DiGraph:
        """
        Building a graph of possible edges.
        Only edges within a time point and to a future time point are allowed in order to build a DBN.
        """
        constraint_graph = nx.DiGraph()
        constraint_graph.add_nodes_from(list(range(n_nodes)))
        n_nodes_per_time = n_nodes // self.timesteps

        for t0 in range(self.timesteps):
            for n0 in range(n_nodes_per_time):
                node_start = t0 * n_nodes_per_time + n0
                for t1 in range(t0, self.timesteps):
                    for n1 in range(n_nodes_per_time):
                        node_end = t1 * n_nodes_per_time + n1
                        if node_start != node_end:
                            constraint_graph.add_edge(node_start, node_end)

        return constraint_graph

    def fit(self, X: np.ndarray, y=None) -> 'DynamicBayesianNetworkAD':
        X = self._preprocess_data(X, True)
        constraint_graph = self._build_contraint_graph(X.shape[1])
        self.bayesian_network = pg.BayesianNetwork.from_samples(X, constraint_graph=constraint_graph)
        self.bayesian_network.bake()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess_data(X, False)
        cut_probs = np.zeros(self.timesteps - 1) + np.nan
        return np.concatenate([-self.bayesian_network.log_probability(X), cut_probs])

    def save(self, path: PathLike):
        self.bayesian_network = self.bayesian_network.to_json()
        joblib.dump(self, path)

    @staticmethod
    def load(path: PathLike) -> 'DynamicBayesianNetworkAD':
        model: DynamicBayesianNetworkAD = joblib.load(path)
        model.bayesian_network = pg.BayesianNetwork.from_json(model.bayesian_network)
        return model
