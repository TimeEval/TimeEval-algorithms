import pandas as pd
import argparse
from dataclasses import dataclass, field
import json
import sys
import pickle
from typing import List

from mtad_gat.model import MTAD_GAT


@dataclass
class CustomParameters:
    mag_window_size: int = 3
    score_window_size: int = 40
    threshold: float = 3
    context_window_size: int = 5
    kernel_size: int = 7
    learning_rate: float = 1e-3
    epochs: int = 1
    batch_size: int = 64
    window_size: int = 20
    gamma: float = 0.8
    latent_size: int = 300
    linear_layer_shape: List[int] = field(default_factory=lambda: [300, 300, 300])
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10
    split: float = 0.8
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataInput).drop(["is_anomaly"], axis=1, errors="ignore")

    @property
    def num_features(self) -> int:
        return len(self.df.columns)

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def save_model(model: MTAD_GAT, args: AlgorithmArgs):
    with open(args.modelOutput, "wb") as f:
        pickle.dump(model, f)


def load_model(args: AlgorithmArgs) -> MTAD_GAT:
    with open(args.modelInput, "rb") as f:
        model = pickle.load(f)
    return model


def train(args: AlgorithmArgs):
    df = args.df
    model = MTAD_GAT(
        mag_window=args.customParameters.mag_window_size,
        score_window=args.customParameters.score_window_size,
        batch_size=args.customParameters.batch_size,
        threshold=args.customParameters.threshold,
        around_window_size=args.customParameters.context_window_size,
        kernel_size=args.customParameters.kernel_size,
        window_size=args.customParameters.window_size,
        gamma=args.customParameters.gamma,
        channel_sizes=args.customParameters.linear_layer_shape,
        latent_size=args.customParameters.latent_size,
        num_features=df.shape[1]-1,
        split=args.customParameters.split,
        early_stopping_patience=args.customParameters.early_stopping_patience,
        early_stopping_delta=args.customParameters.early_stopping_delta
    )
    model.fit(df, args.customParameters.epochs, args.customParameters.learning_rate, args.customParameters.batch_size, callback=lambda m: save_model(m, args))
    save_model(model, args)


def execute(args: AlgorithmArgs):
    df = args.df
    model = load_model(args)
    scores = model.detect(df, args.customParameters.batch_size)
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

    print(args)
    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
