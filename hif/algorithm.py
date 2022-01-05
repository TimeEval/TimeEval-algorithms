import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
import sys
import json
import pickle

import numpy as np
import hif


@dataclass
class CustomParameters:
    n_trees: int = 1024
    max_samples: Optional[float] = None
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> "AlgorithmArgs":
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(
                lambda x: x[0] in custom_parameter_keys,
                args.get("customParameters", {}).items(),
            )
        )
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random

    random.seed(seed)
    np.random.seed(seed)


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        ncols = len(f.readline().split(","))
    r_data = np.genfromtxt(path, skip_header=1, delimiter=",", usecols=range(1, ncols))
    X = r_data[:, :-1]
    Y = r_data[:, -1].astype(bool)
    return X, Y


def seperate_anomalies(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # use anomalies as mask array to seperate normal data from anomalies
    data = X[~Y]
    anomalies = X[Y]
    return data, anomalies


def train(config):
    # load data
    X, Y = load_data(config.dataInput)
    train_data, anomalies = seperate_anomalies(X, Y)

    # build tree on normal data
    n_trees = config.customParameters.n_trees
    if config.customParameters.max_samples:
        sample_size = int(config.customParameters.max_samples * X.shape[0])
    else:
        sample_size = min(256, X.shape[0])
    F = hif.hiForest(train_data, n_trees, sample_size)

    # add anomalies to isolation buckets
    for i in range(anomalies.shape[0]):
        F.addAnomaly(x=anomalies[i], lab=1)
    F.computeAnomalyCentroid()

    # save model
    with open(config.modelOutput, "wb") as f:
        pickle.dump(F, f)


def execute(config):
    # load test data, ignore labels (we want to infer them)
    test_data, _ = load_data(config.dataInput)
    size = test_data.shape[0]

    # load model
    with open(config.modelInput, "rb") as f:
        F: hif.hiForest = pickle.load(f)

    # compute anomaly score for test data points
    scores = np.zeros(size)
    for i in range(size):
        score, _, _, _ = F.computeAggScore(test_data[i])
        scores[i] = score

    # save results
    np.savetxt(config.dataOutput, scores)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    set_random_state(config)

    if config.executionType == "train":
        train(config)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
