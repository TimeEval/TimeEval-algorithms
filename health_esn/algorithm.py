#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import pickle
from typing import Callable
from enum import Enum

from dataclasses import dataclass
from health_esn.model import HealthESN
from scipy.special import expit


class Activation(Enum):
    SIGMOID="sigmoid"
    TANH="tanh"

    def get_fun(self) -> Callable[[np.ndarray], np.ndarray]:
        if self == Activation.SIGMOID:
            return expit
        else: # if self == Activation.TANH
            return np.tanh


@dataclass
class CustomParameters:
    linear_hidden_size: int = 500
    prediction_window_size: int = 20
    connectivity: float = 0.25
    spectral_radius: float = 0.6
    activation: str = Activation.TANH.value
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


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def save(args: AlgorithmArgs, model: HealthESN):
    with open(args.modelOutput, "wb") as f:
        pickle.dump(model, f)


def load(args: AlgorithmArgs) -> HealthESN:
    with open(args.modelOutput, "rb") as f:
        model = pickle.load(f)
    return model


def train(args: AlgorithmArgs):
    ts = args.ts
    health_esn = HealthESN(n_dimensions=ts.shape[1],
                           hidden_units=args.customParameters.linear_hidden_size,
                           window_size=args.customParameters.prediction_window_size,
                           connectivity=args.customParameters.connectivity,
                           spectral_radius=args.customParameters.spectral_radius,
                           activation=Activation(args.customParameters.activation).get_fun(),
                           seed=args.customParameters.random_state)
    health_esn.fit(ts)
    save(args, health_esn)


def execute(args: AlgorithmArgs):
    ts = args.ts
    health_esn = load(args)
    anomaly_scores = health_esn.predict(ts)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    args = AlgorithmArgs.from_sys_args()
    print(f"AlgorithmArgs: {args}")

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"Unknown execution type '{args.executionType}'; expected either 'train' or 'execute'!")
