#!/usr/bin/env python3
import json
import sys

import numpy as np
import eif as iso
from pathlib import Path
from typing import Optional


class Config:
    dataInput: Path
    dataOutput: Path
    executionType: str
    n_trees: int
    max_samples: Optional[float]
    extension_level: int
    limit: int
    random_state: int

    def __init__(self, params):
        self.dataInput = Path(params.get("dataInput", "/data/dataset.csv"))
        self.dataOutput = Path(
            params.get("dataOutput", "/results/anomaly_window_scores.ts")
        )
        self.executionType = params.get("executionType", "execute")
        try:
            customParameters = params["customParameters"]
        except KeyError:
            customParameters = {}
        self.n_trees = customParameters.get("n_trees", 200)
        self.max_samples = customParameters.get("max_samples", None)
        self.extension_level = customParameters.get("extension_level", None)
        self.limit = customParameters.get("limit", None)
        self.random_state = customParameters.get("random_state", 42)


def set_random_state(config) -> None:
    seed = config.random_state
    import random

    random.seed(seed)
    np.random.seed(seed)


def read_data(data_path: Path):
    X = np.genfromtxt(data_path, delimiter=",", skip_header=True)
    X = X[:, 1:-1]
    print("Data")
    print("  dims:", len(X[0]))
    print("  samples:", len(X))
    return X


def create_forest(X: int, config: Config):
    print("Creating forest")
    n_trees = config.n_trees
    if config.max_samples:
        sample_size = int(config.max_samples * X.shape[0])
    else:
        sample_size = min(256, X.shape[0])
    limit = config.limit or int(np.ceil(np.log2(sample_size)))
    extension_level = config.extension_level or X.shape[1] - 1
    forest = iso.iForest(
        X,
        ntrees=n_trees,
        sample_size=sample_size,
        limit=limit,
        ExtensionLevel=extension_level,
    )
    return forest


def execute(config: Config):
    print(iso.__version__)
    set_random_state(config)
    X = read_data(config.dataInput)
    forest = create_forest(X, config)

    print("Computing scores")
    scores = forest.compute_paths(X_in=X)
    print(scores)

    np.savetxt(config.dataOutput, scores, delimiter=",", fmt="%f")
    print(f"Results saved to {config.dataOutput}")


def parse_args():
    print(sys.argv)
    if len(sys.argv) < 2:
        print("No arguments supplied, using default arguments!", file=sys.stderr)
        params = {}
    elif len(sys.argv) > 2:
        print("Wrong number of arguments supplied! Single JSON-String expected!", file=sys.stderr)
        exit(1)
    else:
        params = json.loads(sys.argv[1])
    return Config(params)


if __name__ == "__main__":
    config = parse_args()
    if config.executionType == "train":
        print("Nothing to train.")
    elif config.executionType == "execute":
        execute(config)
    else:
        raise Exception("Invalid Execution type given")
