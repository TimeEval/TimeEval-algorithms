import pandas as pd
import numpy as np
import json
import sys
from dataclasses import dataclass
import argparse

from srcnn.generate_data import generate_data
from srcnn.train import train as train_srcnn
from srcnn.evalue import evaluate


@dataclass
class CustomParameters:
    window_size: int = 128
    step: int = 64
    random_state: int = 42
    num: int = 10
    learning_rate: float = 1e-6
    epochs: int = 1
    batch_size: int = 256
    n_jobs: int = 1
    split: float = 0.8
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10


class AlgorithmArgs(argparse.Namespace):
    @property
    def df(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataInput)
        df = df.iloc[:, [0, 1, df.shape[1]-1]]
        df.columns = ["timestamp", "value", "is_anomaly"]
        return df

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def train(args: AlgorithmArgs):
    data_path = generate_data(args.dataInput,
                              args.customParameters.window_size,
                              args.customParameters.step,
                              args.customParameters.random_state,
                              args.customParameters.num)
    train_srcnn(data_path,
                args.customParameters.window_size,
                args.customParameters.learning_rate,
                args.customParameters.step,
                args.customParameters.random_state,
                False,
                args.modelOutput,
                args.customParameters.epochs,
                args.customParameters.batch_size,
                args.customParameters.n_jobs,
                "sr_cnn",
                args.customParameters.split,
                args.customParameters.early_stopping_delta,
                args.customParameters.early_stopping_patience)


def execute(args: AlgorithmArgs):
    scores = evaluate(args.modelInput, args.df, args.customParameters.window_size)
    scores = np.array(scores[0][1])
    scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    import numpy as np
    import torch
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
