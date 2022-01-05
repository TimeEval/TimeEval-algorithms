import argparse
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
import sys

from encdec_ad.model import EncDecAD


@dataclass
class CustomParameters:
    lstm_layers: int = 1
    split: float = 0.9
    anomaly_window_size: int = 30
    latent_size: int = 40
    batch_size: int = 32
    validation_batch_size: int = 128
    test_batch_size: int = 128
    epochs: int = 50  # bigger for smaller datasets, smaller for bigger datasets
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10
    learning_rate: float = 1e-3
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
    data = args.ts
    model = EncDecAD(input_size=data.shape[1], **asdict(args.customParameters))
    model.fit(data, args.modelOutput)
    model.save(args.modelOutput)


def execute(args: AlgorithmArgs):
    data = args.ts
    model = EncDecAD.load(args.modelInput, input_size=data.shape[1], **asdict(args.customParameters))
    anomaly_scores = model.anomaly_detection(data)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
