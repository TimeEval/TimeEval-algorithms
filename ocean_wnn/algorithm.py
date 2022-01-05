import argparse
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import sys

from ocean_wnn.model import WNN


@dataclass
class CustomParameters:
    train_window_size: int = 20
    hidden_size: int = 100
    batch_size: int = 64
    test_batch_size: int = 256
    epochs: int = 1
    split: float = 0.8
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10
    learning_rate: float = 0.01
    wavelet_a: float = -2.5
    wavelet_k: float = -1.5
    wavelet_wbf: str = "mexican_hat"  # "mexican_hat", "central_symmetric", "morlet"
    wavelet_cs_C: float = 1.75
    threshold_percentile: float = 0.99
    with_threshold: bool = True
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
    model = WNN(window_size=args.customParameters.train_window_size,
                hidden_size=args.customParameters.hidden_size,
                a=args.customParameters.wavelet_a,
                k=args.customParameters.wavelet_k,
                wbf=args.customParameters.wavelet_wbf,
                C=args.customParameters.wavelet_cs_C)
    model.fit(data,
              epochs=args.customParameters.epochs,
              learning_rate=args.customParameters.learning_rate,
              batch_size=args.customParameters.batch_size,
              test_batch_size=args.customParameters.test_batch_size,
              split=args.customParameters.split,
              early_stopping_delta=args.customParameters.early_stopping_delta,
              early_stopping_patience=args.customParameters.early_stopping_patience,
              threshold_percentile=args.customParameters.threshold_percentile,
              model_path=args.modelOutput)
    model.save(args.modelOutput)


def execute(args: AlgorithmArgs):
    data = args.ts
    model = WNN.load(args.modelInput)
    scores, _ = model.detect(data, with_threshold=args.customParameters.with_threshold)
    scores.tofile(args.dataOutput, sep="\n")


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
