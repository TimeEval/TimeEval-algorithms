#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from model import RandomBlackForestAnomalyDetector


@dataclass
class CustomParameters:
    train_window_size: int = 50
    n_estimators: int = 2  # number of forests
    max_features_per_estimator: float = 0.5  # fraction of features per forest
    n_trees: float = 100  # number of trees per forest
    max_features_method: str = "auto"  # "sqrt", "log2"
    bootstrap: bool = True
    max_samples: Optional[float] = None  # fraction of all samples
    # standardize: bool = False  # does not really influence the quality
    random_state: int = 42
    verbose: int = 0
    n_jobs: int = 1
    # the following parameters control the tree size
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1


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
    return data, labels


def train(config: AlgorithmArgs):
    np.random.seed(config.customParameters.random_state)
    data, _ = load_data(config)

    print("Training random forest classifier")
    args = asdict(config.customParameters)
    model = RandomBlackForestAnomalyDetector(**args).fit(data)
    print(f"Saving model to {config.modelOutput}")
    model.save(Path(config.modelOutput))

def execute(config: AlgorithmArgs):
    np.random.seed(config.customParameters.random_state)
    data, _ = load_data(config)
    print(f"Loading model from {config.modelInput}")
    model = RandomBlackForestAnomalyDetector.load(Path(config.modelInput))

    print("Forecasting and calculating errors")
    scores = model.detect(data)
    np.savetxt(config.dataOutput, scores, delimiter=",")
    print(f"Stored anomaly scores at {config.dataOutput}")

    # predictions = model.predict(data)
    # plot(data, predictions, scores, _)


def plot(data, predictions, scores, labels):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler

    # for better visuals, align scores to value range of labels (0, 1)
    scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).reshape(-1)

    fig, axs = plt.subplots(data.shape[1]+1, sharex=True)
    for i in range(data.shape[1]):
        axs[i].plot(data[:, i], label="truth")
        axs[i].plot(predictions[:, i], label="predict")
        axs[i].legend()
    axs[-1].plot(labels, label="label")
    axs[-1].plot(scores, label="score")
    axs[-1].legend()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    set_random_state(config)
    print(f"Config: {config}")

    if config.executionType == "train":
        train(config)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
