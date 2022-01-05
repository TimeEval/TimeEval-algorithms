#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from model import RandomForestAnomalyDetector


@dataclass
class CustomParameters:
    train_window_size: int = 100
    # standardize: bool = False  # does not really influence the quality
    n_estimators: int = 100
    learning_rate: float = 0.1
    booster: str = "gbtree"  # "gblinear", "dart"
    tree_method: str = "auto"  # "exact", "approx", "hist"
    n_trees: int = 1
    max_depth: Optional[int] = None
    max_samples: Optional[float] = None  # Subsample ratio of the training instance
    colsample_bytree: Optional[float] = None  # Subsample ratio of columns when constructing each tree.
    colsample_bylevel: Optional[float] = None  # Subsample ratio of columns for each level.
    colsample_bynode: Optional[float] = None  # Subsample ratio of columns for each split.
    random_state: int = 42
    verbose: int = 0
    n_jobs: int = 1


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
    data = df.iloc[:, 1:2].values
    labels = df.iloc[:, -1].values
    return data, labels


def train(config: AlgorithmArgs):
    data, _ = load_data(config)

    print("Training XGBoost regressor")
    model = RandomForestAnomalyDetector(**asdict(config.customParameters)).fit(data)
    print(f"Saving model to {config.modelOutput}")
    model.save(Path(config.modelOutput))


def execute(config: AlgorithmArgs):
    data, _ = load_data(config)
    print(f"Loading model from {config.modelInput}")
    model = RandomForestAnomalyDetector.load(Path(config.modelInput))

    print("Forecasting and calculating errors")
    scores = model.detect(data)
    np.savetxt(config.dataOutput, scores, delimiter=",")
    print(f"Stored anomaly scores at {config.dataOutput}")

    # predictions = model.predict(data)
    # plot(model, data, predictions, scores)


def plot(model, data, predictions, scores):
    import matplotlib.pyplot as plt

    if model.preprocessor.scaler:
        print("Re-scaling scores for better visual representation")
        scores = model.preprocessor.scaler.inverse_transform(scores)
        scores = scores - model.preprocessor.scaler.mean_

    plt.Figure()
    plt.plot(data, label="truth")
    plt.plot(predictions, label="predict")
    plt.plot(scores, label="score")
    plt.legend()
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
