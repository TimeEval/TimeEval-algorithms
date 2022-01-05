#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass, asdict
from deepnap.model import DeepNAP


@dataclass
class CustomParameters:
    anomaly_window_size: int = 15
    partial_sequence_length: int = 3
    lstm_layers: int = 2
    rnn_hidden_size: int = 200
    dropout: float = 0.5
    linear_hidden_size: int = 100
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 0.001
    split: float = 0.8
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10
    validation_batch_size: int = 256
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
    input_size = ts.shape[1]
    deepnap = DeepNAP(input_size=input_size, **asdict(args.customParameters))
    deepnap.fit(ts, args)
    deepnap.save(args)


def execute(args: AlgorithmArgs):
    ts = args.ts
    deepnap = DeepNAP.load(args)
    anomaly_scores = deepnap.anomaly_detection(ts)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)
    print(f"AlgorithmArgs: {args}")

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"Unknown execution type '{args.executionType}'; expected either 'train' or 'execute'!")
