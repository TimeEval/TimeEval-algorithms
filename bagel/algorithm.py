import pandas as pd
import numpy as np

import sys
import json
import argparse
from typing import List
from dataclasses import dataclass, field
import pickle

from bagel.model import DonutX
from bagel.kpi_series import KPISeries


@dataclass
class CustomParameters:
    window_size: int = 120
    latent_size: int = 8
    hidden_layer_shape: List[int] = field(default_factory=lambda: [100, 100])
    dropout: float = 0.1
    cuda: bool = False
    epochs: int = 50
    batch_size: int = 128
    split: float = 0.8
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.05
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        dataset = pd.read_csv(self.dataInput)
        return dataset.values[:, 1:2]

    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataInput, parse_dates=["timestamp"], infer_datetime_format=True)

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def prepare_data(args: AlgorithmArgs, split_first_half: bool, execute: bool) -> KPISeries:
    df = args.df
    if not execute:
        split_at = int(len(df) * args.customParameters.split)
        df = df.iloc[:split_at] if split_first_half else df.iloc[split_at:]

    kpi = KPISeries(
        value=df.iloc[:, 1],
        timestamp=df.timestamp,
        label=df.is_anomaly,
        name='sample_data',
    )

    kpi = kpi.normalize()
    return kpi


def train(args: AlgorithmArgs, kpi: KPISeries):
    def save_model():
        with open(args.modelOutput, "wb") as f:
            pickle.dump(model, f)

    cp = args.customParameters
    valid_kpi = prepare_data(args, False, False)

    model = DonutX(
        window_size=cp.window_size,
        latent_dims=cp.latent_size,
        network_size=cp.hidden_layer_shape,
        batch_size=cp.batch_size,
        condition_dropout_left_rate=1-cp.dropout,
        cuda=cp.cuda,
        max_epoch=cp.epochs,
        early_stopping_delta=cp.early_stopping_delta,
        early_stopping_patience=cp.early_stopping_patience
    )
    try:
        model.fit(kpi.label_sampling(0.), valid_kpi=valid_kpi.label_sampling(0.), callbacks=[(lambda i, _e, _l: save_model() if i else None)])
        save_model()
    except StopIteration:
        # Silently fail if the training stopped early
        pass


def execute(args: AlgorithmArgs, kpi: KPISeries):
    with open(args.modelInput, "rb") as f:
        model = pickle.load(f)

    y_prob: np.ndarray = model.predict(kpi)
    y_prob.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    kpi = prepare_data(args, True, args.executionType == "execute")

    if args.executionType == "train":
        train(args, kpi)
    elif args.executionType == "execute":
        execute(args, kpi)
    else:
        ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
