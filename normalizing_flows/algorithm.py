import argparse
from dataclasses import dataclass, asdict, field
import json
import numpy as np
import pandas as pd
import sys
from typing import List
from pathlib import Path

from normalizing_flows.model import NormalizingFlow


@dataclass
class CustomParameters:
    n_hidden_features_factor: float = 1.0
    hidden_layer_shape: List[int] = field(default_factory=lambda: [100, 100])
    window_size: int = 20
    split: float = 0.9
    epochs: int = 1
    batch_size: int = 64
    test_batch_size: int = 128
    teacher_epochs: int = 1
    distillation_iterations: int = 1
    percentile: float = 0.05
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.05
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        return self.df.iloc[:, 1:-1].values

    @property
    def targets(self) -> np.ndarray:
        return self.df.iloc[:, -1].values

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
    targets = args.targets
    kwargs = asdict(args.customParameters)
    kwargs["n_features"] = data.shape[1] * kwargs["window_size"]
    kwargs["n_hidden_features"] = int(kwargs["n_features"] * kwargs.pop("n_hidden_features_factor"))
    del kwargs["random_state"]
    model = NormalizingFlow(**kwargs).fit(data, targets, model_path=Path(args.modelOutput))
    model.save(Path(args.modelOutput))


def execute(args: AlgorithmArgs):
    data = args.ts
    model = NormalizingFlow.load(Path(args.modelInput))
    scores = model.detect(data)
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
