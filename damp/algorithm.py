#!/usr/bin/env python3
from typing import Optional
import argparse
import json
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from damp import DAMP


@dataclass
class CustomParameters:
    anomaly_window_size: int = 50
    n_init_train: int = 100
    max_lag: Optional[int] = None
    lookahead: Optional[int] = None
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        return self.df.iloc[:, 1:-1].values

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

    damp = DAMP(m=config.customParameters.anomaly_window_size,
                sp_index=config.customParameters.n_init_train,
                x_lag=config.customParameters.max_lag,
                lookahead=config.customParameters.lookahead)
    mp = damp.fit_transform(data)

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
