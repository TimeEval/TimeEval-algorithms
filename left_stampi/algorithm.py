#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from stumpy import stumpi


@dataclass
class CustomParameters:
    anomaly_window_size: int = 50
    n_init_train: int = 100
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        return self.df.values[:, 1].astype(float)

    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataInput)

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


def main(config: AlgorithmArgs):
    set_random_state(config)
    data = config.ts
    warmup = config.customParameters.n_init_train

    ws = config.customParameters.anomaly_window_size
    if ws > warmup:
        print(f"WARN: anomaly_window_size is larger than n_init_train. Dynamically fixing it by setting anomaly_window_size to n_init_train={warmup}")
        ws = warmup
    if ws < 3:
        print("WARN: anomaly_window_size must be at least 3. Dynamically fixing it by setting anomaly_window_size to 3")
        ws = 3

    stream = stumpi(data[:warmup], m=ws, egress=False)
    for point in data[warmup:]:
        stream.update(point)

    mp = stream.left_P_
    mp[:warmup] = 0

    np.savetxt(config.dataOutput, mp, delimiter=",")


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
