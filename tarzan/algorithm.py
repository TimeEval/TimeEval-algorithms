import numpy as np
import pandas as pd
import json
import sys
from dataclasses import dataclass
import argparse
import pickle

import tarzan.TARZAN as TARZAN


@dataclass
class CustomParameters:
    anomaly_window_size: int = 20
    alphabet_size: int = 4
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        return self.df.values[:, 1]

    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataInput)

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


# the training and execution happens in one
def train(args: AlgorithmArgs):
    data = args.ts
    with open(args.modelOutput, "wb") as f:
        pickle.dump(data, f)


def execute(args: AlgorithmArgs):
    data = args.ts
    with open(args.modelInput, "rb") as f:
        train_data = pickle.load(f)
    scores = TARZAN.TARZAN(train_data, data, args.customParameters.anomaly_window_size, args.customParameters.alphabet_size)
    np.array(scores).tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    print(f"Configuration: {args}")
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
