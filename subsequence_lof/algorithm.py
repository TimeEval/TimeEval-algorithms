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
    use_column_index: int = 0


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
    column_index = 0
    if config.customParameters.use_column_index is not None:
        column_index = config.customParameters.use_column_index
    max_column_index = df.shape[1] - 3
    if column_index > max_column_index:
        print(f"Selected column index {column_index} is out of bounds (columns = {df.columns.values}; "
              f"max index = {max_column_index} [column '{df.columns[max_column_index + 1]}'])! "
              "Using last channel!", file=sys.stderr)
        column_index = max_column_index
    # jump over index column (timestamp)
    column_index += 1

    data = df.iloc[:, column_index].values
    labels = df.iloc[:, -1].values
    contamination = labels.sum() / len(labels)
    # Use smallest positive float as contamination if there are no anomalies in dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination
    return data, contamination


def main(config: AlgorithmArgs):
    set_random_state(config)
    data, contamination = load_data(config)

    # preprocess data
    data = sliding_window_view(data, window_shape=config.customParameters.window_size)

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
