import numpy as np
import pandas as pd
import json
import sys
from dataclasses import dataclass
import argparse
import pickle

from robust_pca.model import AnomalyDetector


@dataclass
class CustomParameters:
    max_iter: int = 1000
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


def save(args: AlgorithmArgs, model: AnomalyDetector):
    with open(args.modelOutput, "wb") as f:
        pickle.dump(model, f)


def load(args: AlgorithmArgs) -> AnomalyDetector:
    with open(args.modelInput, "rb") as f:
        model = pickle.load(f)
    return model


def train(args: AlgorithmArgs):
    data = args.ts
    detector = AnomalyDetector(args.customParameters.max_iter)
    detector.fit(data)
    save(args, detector)


def execute(args: AlgorithmArgs):
    data = args.ts
    detector = load(args)
    anomaly_scores = detector.detect(data)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
