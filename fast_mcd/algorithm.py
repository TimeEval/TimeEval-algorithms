import argparse
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import sys
from typing import Optional

from sklearn.covariance import MinCovDet
import joblib


@dataclass
class CustomParameters:
    store_precision: bool = True
    support_fraction: Optional[float] = None
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


def train(args: AlgorithmArgs):
    set_random_state(args)
    ts = args.ts
    model = MinCovDet(
        store_precision=args.customParameters.store_precision,
        assume_centered=False,
        support_fraction=args.customParameters.support_fraction,
        random_state=args.customParameters.random_state
    )
    model.fit(ts)
    joblib.dump(model, args.modelOutput)


def execute(args: AlgorithmArgs):
    ts = args.ts
    model = joblib.load(args.modelInput)
    scores = model.mahalanobis(ts)
    scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
