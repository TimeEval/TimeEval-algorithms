#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
from typing import List

from dataclasses import dataclass, asdict, field
from mscred.model import MSCRED


@dataclass
class CustomParameters:
    windows: List[int] = field(default_factory=lambda: [10, 30, 60])
    gap_time: int = 10
    window_size: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 1
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.05
    split: float = 0.8
    test_batch_size: int = 256
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
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def train(args: AlgorithmArgs):
    ts = args.ts
    params = asdict(args.customParameters)
    del params["random_state"]
    mscred = MSCRED(n_dimensions=ts.shape[1], **params)
    mscred.fit(ts, args)
    mscred.save(args)


def execute(args: AlgorithmArgs):
    ts = args.ts
    mscred = MSCRED.load(args)
    anomaly_scores = mscred.detect(ts)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    args = AlgorithmArgs.from_sys_args()
    print(f"AlgorithmArgs: {args}")
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"Unknown execution type '{args.executionType}'; expected either 'train' or 'execute'!")
