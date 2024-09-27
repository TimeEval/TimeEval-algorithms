#!/usr/bin/env python3
import json
import sys
from argparse import Namespace

import pandas as pd
import numpy as np

from dataclasses import dataclass
from SAND import SAND


@dataclass
class CustomParameters:
    anomaly_window_size: int = 75
    n_clusters: int = 6
    n_init_train: int = 2000
    iter_batch_size: int = 500
    alpha: float = 0.5
    random_state: int = 42
    use_column_index: int = 0


class AlgorithmArgs(Namespace):
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

    return df.iloc[:, column_index].values


def main(config: AlgorithmArgs):
    set_random_state(config)
    data = load_data(config)
    scores = np.full_like(data, fill_value=np.nan)
    print(f"Data shape: {data.shape}")

    # empirically shown best value
    subsequence_length = 3 * config.customParameters.anomaly_window_size
    # Take subsequence every 'overlaping_rate' points
    # Change it to 1 for completely overlapping subsequences
    # Change it to 'subsequence_length' for non-overlapping subsequences
    # Change it to 'subsequence_length//4' for non-trivial matching subsequences
    # --> use non-trivial matching, but guard against
    overlaping_rate = max(subsequence_length//4, 1)
    init_size = config.customParameters.n_init_train
    batch_size = config.customParameters.iter_batch_size

    print(f"Initializing on first {init_size} points")
    sand = SAND(data,
        k=config.customParameters.n_clusters,
        init_length=init_size,
        batch_size=batch_size,
        pattern_length=config.customParameters.anomaly_window_size,
        subsequence_length=subsequence_length,
        alpha=config.customParameters.alpha,
        overlaping_rate=overlaping_rate,
    )
    sand.initialize()

    # patch sand to compute scores for all initial points (not only the last batch_size)
    sand.batch_size = init_size
    scores[:init_size] = sand.compute_score()

    # remove sand patch: reset batch size
    sand.batch_size = batch_size

    i = 0
    while sand.current_time < len(data):
        start = init_size + i*batch_size
        end = min(init_size + (i+1)*batch_size, len(data))

        if start+subsequence_length >= len(data):
            print(f"Last batch {i} is too small ({end-start} < {subsequence_length}), skipping")
            break

        print(f"Computing batch {i} ({start}-{end})")
        sand.run_next_batch()
        scores[start:end] = sand.compute_score()
        i += 1

    print("Storing scores")
    print(f"Scores shape: {scores.shape}")
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
