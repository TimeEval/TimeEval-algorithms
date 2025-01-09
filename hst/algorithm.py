#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from numpy.lib.stride_tricks import sliding_window_view

from hst import HalfSpaceTrees
from river import compose, preprocessing



@dataclass
class CustomParameters:
    n_trees: int = 10
    height: int = 8 
    window_size: int = 250
    random_state: int = 42


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

def read_csv_in_batches(filepath, batch_size):
    iterator = pd.read_csv(filepath, chunksize=batch_size)
    
    for batch in iterator:
        yield batch["value"].values


def main(config: AlgorithmArgs):
    batch_size = 1024
    subsequence_length = 20

    set_random_state(config)

    model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            HalfSpaceTrees(n_trees=config.customParameters.n_trees, height=config.customParameters.height, window_size=config.customParameters.window_size,
            seed=config.customParameters.random_state)
    )

    scores = np.zeros(batch_size)

    for batch in read_csv_in_batches(config.dataInput, batch_size):

        subsequences = sliding_window_view(batch, window_shape=subsequence_length)
        features = {i: 0 for i in range(subsequence_length)}

        for i, subsequence in enumerate(subsequences):
            for j, value in enumerate(subsequence):
                features[j] = value
            model.learn_one(features)
            scores[i] = model.score_one(features)


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
