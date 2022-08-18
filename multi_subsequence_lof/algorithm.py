#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pyod.models.lof import LOF
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class CustomParameters:
    window_size: int = 100
    n_neighbors: int = 20
    leaf_size: int = 30
    distance_metric_order: int = 2
    n_jobs: int = 1
    algorithm: str = "auto"  # using default is fine
    distance_metric: str = "minkowski"  # using default is fine
    random_state: int = 42
    dim_aggregation_method: str = "concat"  # or "sum"


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_data(config: AlgorithmArgs) -> np.ndarray:
    df = pd.read_csv(config.dataInput)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    contamination = labels.sum() / len(labels)
    # Use smallest positive float as contamination if there are no anomalies in dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination
    return data, contamination


def main(config: AlgorithmArgs):
    set_random_state(config)
    data, contamination = load_data(config)

    # preprocess data
    n_dims = data.shape[1]
    if config.customParameters.dim_aggregation_method == "sum":
        # move all channels above zero (minimum of each curve is 0)
        if len(data.shape) > 1:
            data = data - data.min()
        data = sliding_window_view(data, window_shape=config.customParameters.window_size, axis=0)
        data = data.sum(axis=1)
    else:
        data = sliding_window_view(data, window_shape=config.customParameters.window_size, axis=0)
        data = data.reshape((-1, config.customParameters.window_size*n_dims))

    clf = LOF(
        contamination=contamination,
        n_neighbors=config.customParameters.n_neighbors,
        leaf_size=config.customParameters.leaf_size,
        n_jobs=config.customParameters.n_jobs,
        algorithm=config.customParameters.algorithm,
        metric=config.customParameters.distance_metric,
        metric_params=None,
        p=config.customParameters.distance_metric_order,
    )
    clf.fit(data)
    scores = clf.decision_scores_
    np.savetxt(config.dataOutput, scores, delimiter=",")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    print(f"Config: {config}")

    if config.executionType == "train":
        print("Nothing to train, finished!")
    elif config.executionType == "execute":
        main(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
